from scipy.interpolate import interp1d
import math
import numpy as np
from ADRpy import atmospheres as at
import ussa1976
from specs import *

# Constants
g = 9.81  # m/s²
atm = at.Atmosphere()

# functions

def kmh_to_ms(v_kmh):
    return v_kmh / 3.6

def estimate_cruise_sfc_from_dicts(power_vs_rpm_dict, sfc_vs_rpm_dict, cruise_power):
    # Convert keys and values to sorted numeric arrays
    rpm_power = sorted((int(k), v) for k, v in power_vs_rpm_dict.items())
    rpm_sfc = sorted((int(k), v) for k, v in sfc_vs_rpm_dict.items())

    rpm_vals_p, power_vals = zip(*rpm_power)
    rpm_vals_sfc, sfc_vals = zip(*rpm_sfc)

    # Interpolate to find cruise RPM
    rpm_from_power = interp1d(power_vals, rpm_vals_p, kind='linear', fill_value='extrapolate')
    rpm_cruise = float(rpm_from_power(cruise_power))

    # Interpolate to find SFC at cruise RPM
    sfc_from_rpm = interp1d(rpm_vals_sfc, sfc_vals, kind='linear', fill_value='extrapolate')
    sfc_cruise = float(sfc_from_rpm(rpm_cruise))

    return {
        "rpm_cruise": rpm_cruise,
        "sfc_cruise": sfc_cruise
    }


# --- 1. Get air density at cruise altitude ---
def get_air_density(altitude_m):
    return atm.airdens_kgpm3(altitude_m)


def get_dynamics_viscosity_mu(altitude_meters):
    """
    Calculates the dynamic viscosity of air at a given altitude using the
    U.S. Standard Atmosphere 1976 model.

    Args:
        altitude_meters (float): The geopotential altitude in meters.

    Returns:
        float: The dynamic viscosity of air in Pa*s at the specified altitude.
    """
    # Compute the U.S. Standard Atmosphere 1976 model for the given altitude
    ds = ussa1976.compute(np.array([altitude_meters]))
    
    # Extract the dynamic viscosity (mu) from the computed dataset
    dynamic_viscosity = ds['mu'].values[0]  / 1e12
    
    return dynamic_viscosity

def calculate_reynolds_number(velocity_kmh, chord_m, altitude_m=0):
    rho = get_air_density(altitude_m)  # from helper_functions
    mu = get_dynamics_viscosity_mu(altitude_m)  # kg/m·s, dynamic viscosity at 15°C
    v_ms = kmh_to_ms(velocity_kmh)
    Re = (rho * v_ms * chord_m) / mu

    return Re

# --- 3. Estimate Drag Coefficient using parabolic drag polar ---
def calc_cd_total(cd0, cl, aspect_ratio, e):
    k = 1 / (np.pi * e * aspect_ratio)
    return cd0 + k * cl**2


def calculate_cg(weights_dict_input, positions_dict):

    weights_dict = weights_dict_input.copy()

    del weights_dict["horizontal_tail"]
    del weights_dict["vertical_tail"]
    
    numerator = sum(weights_dict[k] * positions_dict[k] for k in weights_dict)
    denominator = sum(weights_dict.values())
    return numerator / denominator


def estimate_component_positions(current_values, hard_constraints, assumed_and_set, weights_dict_kg_no_fuel, internal_payload_x = None):

    cg_estimate = current_values[f"cruiseout_cg_from_nose_m"]

    tail_local_cg_ht_z = weights_dict_kg_no_fuel["horizontal_tail"] * (current_values["h_tail_chord_m"] * assumed_and_set["wing_airfoil_thickness_ratio"]) / 2
    tail_local_cg_vt_z = weights_dict_kg_no_fuel["vertical_tail"] * current_values["v_tail_span_m"] / 2
    tail_local_cg_z = (tail_local_cg_ht_z + tail_local_cg_vt_z) / weights_dict_kg_no_fuel["tails"]

    if internal_payload_x is not None:
        internal_payload_x_position = internal_payload_x
    else:
        internal_payload_x_position = (0.5*hard_constraints["internal_payload_length"]) + 1.5

    wing_vertical_position = current_values['high_wing_offset_m'] + (current_values['fuselage_body_height_m']/2)
    full_dict = {
        "fuselage": (0.5 * current_values["fuselage_body_length_m"], 
                     0.5*current_values["fuselage_body_height_m"]),
        "wing": (current_values["wing_le_position_m"] + 0.45 * current_values["chord_m"], 
                 wing_vertical_position),
        "tails": (current_values["x_ht_le_m"] + assumed_and_set["tail_mass_cg_from_le_coeff"]  * current_values["h_tail_chord_m"], 
                  tail_local_cg_z + current_values["tail_boom_pylon_height_m"]),
        # "tails": 0.9*fuselage_length,
        "engine": (current_values["fuselage_body_length_m"] * 0.0 + 0.5*engine_specs["length"], 
                   0.5*current_values["fuselage_body_height_m"]),
        "propeller": (-0.2, 
                      0.5*current_values["fuselage_body_height_m"] ),

        # "internal_payload": (current_values["fuselage_body_length_m"] - 1 - (0.5*hard_constraints["internal_payload_length"]), 
        "internal_payload": (internal_payload_x_position, 
                             0.35*current_values["fuselage_body_height_m"]),
        "wing_payload": (current_values["wing_le_position_m"] + 0.5 * current_values["chord_m"], 
                         -0.2*current_values["fuselage_body_height_m"]),
        "wing_fuel": (current_values["wing_le_position_m"] + 0.45 * current_values["chord_m"], 
                      assumed_and_set["wing_airfoil_thickness_ratio"] * current_values["chord_m"]),
        "fuselage_fuel": (current_values["wing_le_position_m"] - (0.5*assumed_and_set["fuselage_fuel_tank_length"]), 
                          0.35*current_values["fuselage_body_height_m"]),
        "avionics": (current_values["fuselage_body_length_m"] * 0.8, 
                     0.2*current_values["fuselage_body_height_m"]),
        "landing_gear": (0.7 * current_values["fuselage_body_length_m"], 
                         -0.1 * current_values["fuselage_body_height_m"]),
        "misc": (cg_estimate, 
                 0.5 * current_values["fuselage_body_height_m"]),
    }

    cg_x_dict = {k: v[0] for k, v in full_dict.items()}
    cg_z_dict = {k: v[1] for k, v in full_dict.items()}

    return (cg_x_dict, cg_z_dict)

def calculate_eta_h(current_values, phase='cruiseout'):
    """
    Estimate elevator effectiveness factor η_h.
    Computes tail arm from wing AC to tail AC internally using current_values.
    
    Parameters:
    - current_values (dict): Design dictionary (e.g., predrop)
    - phase (str): Flight phase ('cruise', 'loiter', 'takeoff')
    
    Returns:
    - eta_h (float): Elevator effectiveness (typically 0.6–0.9)
    """
    # 1. Wing Aerodynamic Center (AC) at 25% MAC
    wing_chord = current_values["chord_m"]
    wing_ac = current_values[f"{phase}_x_ac_w_m"]

    # 2. Horizontal Tail Aerodynamic Center (AC)
    # Tail LE is located at: cg + tail_arm - ht_chord*0.25 (assuming tail measured from cg)
    # But in your code: tail_arm is from CG to tail LE → so:

    ht_ac = current_values[f"{phase}_x_ht_ac_m"]  # HT AC

    # 3. Moment Arm from Wing AC to HT AC
    tail_arm_ac_to_ac_m = ht_ac - wing_ac

    # 4. Non-dimensional tail length (l_H / c̄)
    l_h_bar = tail_arm_ac_to_ac_m / wing_chord

    # 5. Estimate downwash effect based on l_h_bar
    if l_h_bar < 3:
        downwash_factor = 0.7   # strong downwash
    elif l_h_bar < 5:
        downwash_factor = 0.85  # moderate
    else:
        downwash_factor = 0.95  # weak

    # 6. Base effectiveness
    base_eta = 0.9

    # 7. Adjust for flight phase (higher downwash in high-lift conditions)
    if phase in ['takeoff', 'loiter']:
        downwash_factor *= 0.92

    # 8. Final eta_h
    eta_h = base_eta * downwash_factor

    # Only clamp to prevent non-physical values, not to force optimism
    return eta_h  # realistic lower bound


# Calculate Usable Fuel Tank Volume for Wing Using Selig S1223 Airfoil
def load_airfoil_coordinates(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coords = [list(map(float, line.strip().split())) for line in lines if len(line.strip().split()) == 2]
    return np.array(coords)

def wing_fuel_tank_volume(file_path, iterable_constraints, front_spar_pos, rear_spar_pos, reduction_margin=0.9):
    # Calculate the usable cross-sectional area of the tank
    
# --- Parameters ---
    chord_length = iterable_constraints["chord_m"]  # meters
    span_length = 0.25 * iterable_constraints["wing_span_m"]  # meters (length of wing tank section)

    # --- Load Airfoil Data ---


    # Example: Replace 's1223.dat' with your actual airfoil coordinate file
    airfoil_coords = load_airfoil_coordinates(file_path)

    x_coords = airfoil_coords[:, 0] * chord_length
    y_coords = airfoil_coords[:, 1] * chord_length

    # Separate upper and lower surfaces
    upper_surface = airfoil_coords[:len(airfoil_coords)//2]
    lower_surface = airfoil_coords[len(airfoil_coords)//2:]

    x_upper = upper_surface[:, 0] * chord_length
    y_upper = upper_surface[:, 1] * chord_length

    x_lower = lower_surface[:, 0] * chord_length
    y_lower = lower_surface[:, 1] * chord_length

    # Filter for Tank Region Between Spars
    mask_upper = (x_upper >= front_spar_pos * chord_length) & (x_upper <= rear_spar_pos * chord_length)
    mask_lower = (x_lower >= front_spar_pos * chord_length) & (x_lower <= rear_spar_pos * chord_length)

    x_tank_upper = x_upper[mask_upper]
    y_tank_upper = y_upper[mask_upper]

    x_tank_lower = x_lower[mask_lower]
    y_tank_lower = y_lower[mask_lower]

    # Ensure arrays are the same size by interpolating the lower surface to match the upper x values
    if len(x_tank_upper) != len(x_tank_lower):
        y_tank_lower = np.interp(x_tank_upper, x_tank_lower, y_tank_lower)
        x_tank_lower = x_tank_upper  # Align x arrays

    # Compute Usable Cross-Sectional Area
    area_upper = np.trapz(y_tank_upper - 0.02, x_tank_upper)
    area_lower = np.trapz(y_tank_lower + 0.02, x_tank_lower)

    usable_cross_section_area = area_upper - area_lower  # In m²

    # Compute Total Tank Volume
    tank_volume_m3 = abs(usable_cross_section_area) * span_length
    return reduction_margin*tank_volume_m3
    

def calculate_Cn_beta_phase(current_values, hard_constraints, vt_airfoil_data, phase):
    """
    Compute Cn_beta for a given phase.
    """
    # Geometry
    S_wing = current_values["wing_area_m2"]
    b_wing = current_values["wing_span_m"]
    S_vt = current_values["vertical_tail_area_m2"]
    l_v = current_values["tail_arm_m"]
    c_vt = current_values["v_tail_chord_m"]
    cg_nose = current_values[f"{phase}_cg_from_nose_m"]
    wing_le_position = current_values["wing_le_position_m"]
    chord = current_values["chord_m"]

    # VT AC position
    vt_ac = cg_nose + l_v + 0.25 * c_vt
    wing_ac = wing_le_position + 0.25 * chord
    l_v_ac = vt_ac - wing_ac  # moment arm

    # Vertical tail volume coefficient
    Vv = (S_vt * l_v_ac) / (S_wing * b_wing)

    # Flight condition
    alt = hard_constraints["cruise_altitude_m"]
    speed_kmh = current_values[f"{phase}_speed_kmh"]
    Re_vt = calculate_reynolds_number(speed_kmh, c_vt, alt)

    # Interpolate airfoil data at Re

    airfoil_at_Re = vt_airfoil_data  # fallback

    cl_alpha_vt_per_deg = get_cl_alpha_at(airfoil_at_Re, 0, "CL",delta=0.1)
    cl_alpha_vt_per_rad = cl_alpha_vt_per_deg * (180 / np.pi)

    # Downwash/shielding factor
    eta_v = calculate_eta_h(current_values, phase)  # assume clean airflow; adjust if needed

    # Cn_beta from vertical tail
    Cn_beta_vt = Vv * cl_alpha_vt_per_rad * eta_v

    # Fuselage contribution (negative)
    fuselage_length = current_values["fuselage_body_length_m"]
    nose_ahead_cg = cg_nose
    Cn_beta_fus = -0.003 * nose_ahead_cg  # empirical per meter

    # Total
    Cn_beta = Cn_beta_vt + Cn_beta_fus

    return {
        f"{phase}_Cn_beta": Cn_beta,
        f"{phase}_Re_vt": Re_vt,
    }

def calculate_Cl_beta_phase(current_values, assumed_and_set, phase):
    """
    Compute Cl_beta for a given phase.
    """
    # Geometry
    b_wing = current_values["wing_span_m"]
    S_wing = current_values["wing_area_m2"]
    AR = assumed_and_set["aspect_ratio"]

    # Design parameters (from assumed_and_set or defaults)
    dihedral_deg = assumed_and_set.get("dihedral_deg", 0.0)
    wing_sweep_deg = assumed_and_set.get("wing_sweep_deg", 0.0)
    high_wing_offset_m = current_values.get("high_wing_offset_m", 0.0)

    # Flight condition
    cl = current_values[f"{phase}_cl"]

    # Dihedral effect (empirical)
    Cl_beta_dihedral = -0.02 * dihedral_deg * (1 + 0.2 * cl) # per degree dihedral

    # Sweep contribution (approx: 1° sweep ≈ 0.7° dihedral)
    Cl_beta_sweep = -0.014 * wing_sweep_deg

    # High-wing offset (pendulum effect)
    if high_wing_offset_m > 0:
        Cl_beta_high_wing = -0.002 * high_wing_offset_m * (b_wing / 2)  # approx
    else:
        Cl_beta_high_wing = 0.0

    # Total Cl_beta
    Cl_beta = Cl_beta_dihedral + Cl_beta_sweep + Cl_beta_high_wing

    return {
        f"{phase}_Cl_beta": Cl_beta,
    }


def size_fuselage_diameter_as_per_payload(internal_payload_count, internal_payload_diameter):
    n = internal_payload_count     # number of cylinders
    d = internal_payload_diameter  # cylinder diameter (m)
    s = 0.02               # spacing between cylinders (m)
    t = 0.015              # carousel mechanism radial thickness (m)

    # 1. Effective diameter
    d_eff = d + s

    # 2. Center-to-center radius
    r_center = d_eff / (2 * math.sin(math.pi / n))

    # 3. Outer radius including half the effective diameter
    R_cylinders = r_center + d_eff/2

    # 4. Add carousel thickness
    R_total = R_cylinders + t

    # 5. Required fuselage diameter
    D_required = 2 * R_total

    return D_required + 0.03

def initial_size_fuselage(hard_constraints, assumed_and_set, iterable_constraints, fuselage_tank_in_wing_root=True):

    nose_for_ballast_and_aerodynamics_length = 0.5
    ## fuel

    # fuel_volume = (iterable_constraints["fuel_kg"] / assumed_and_set["fuel_density_kgL"]) / 1000

    # fuselage_cross_section_area = 0.6*0.45 # sized around engine cross section
    # fuselage_length_for_fuel = 0.03 + (fuel_volume / fuselage_cross_section_area)
    fuselage_length_for_fuel = assumed_and_set["fuselage_fuel_tank_length"]
    
    # ## payload

    internal_payload_length = hard_constraints["internal_payload_length"]

    
    ## avionics (computers, sensors, camera)

    fuselage_length_for_avionics = 0.7 # m 

    ## wing root requirements

    fuselage_length_for_wing_root = 1.2 * iterable_constraints["chord_m"] # this would include the center wing box

    ## powerplant

    fuselage_length_for_engine = 1 # engine length is 0.7 , added 0.3 for filters and controls

    if fuselage_tank_in_wing_root:
        total_fuselage_length = nose_for_ballast_and_aerodynamics_length + internal_payload_length + fuselage_length_for_avionics + fuselage_length_for_wing_root + fuselage_length_for_engine
    else:
        total_fuselage_length = nose_for_ballast_and_aerodynamics_length + fuselage_length_for_fuel + internal_payload_length + fuselage_length_for_avionics + fuselage_length_for_wing_root + fuselage_length_for_engine

    return total_fuselage_length


def get_fuselage_mass(fuselage_length, fuselage_width, fuselage_height, assumed_and_set):
    skin_thickness = sandwich_specs["fuselage"]["total_thickness_m"] - sandwich_specs["fuselage"]["core_thickness_m"]
    skin_cross_sectional_cirfumferance = (fuselage_width + fuselage_height)
    skin_cross_sectional_area = skin_cross_sectional_cirfumferance * skin_thickness
    total_skin_volume = fuselage_length * skin_cross_sectional_area
    skin_density = assumed_and_set["gfrp_density_kgm3"]
    total_skin_mass = total_skin_volume * skin_density
    
    core_thickness = sandwich_specs["fuselage"]["core_thickness_m"]
    core_cross_sectional_cirfumferance = 2 * (fuselage_width + fuselage_height)
    core_cross_sectional_area = core_cross_sectional_cirfumferance * core_thickness
    total_core_volume = fuselage_length * core_cross_sectional_area
    core_density = assumed_and_set["core_density_kgm3"]
    total_core_mass = total_core_volume * core_density

    total_fuselage_mass = total_skin_mass + total_core_mass

    return total_fuselage_mass

def airfoil_perimeter_length(x, y, chord):
    """
    Compute total surface length of an airfoil given x and y coordinates.

    Parameters:
        x (array-like): x coordinates of the airfoil surface
        y (array-like): y coordinates of the airfoil surface

    Returns:
        float: total length along the airfoil perimeter
    """
    x = np.array(x)
    y = np.array(y)

    dx = np.diff(x)
    dy = np.diff(y)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    total_length = np.sum(segment_lengths) * chord
    
    return total_length

def _get_wing_or_tail_mass(span, cross_section_perimeter_length, tail_or_wing, assumed_and_set):

    
    skin_thickness = sandwich_specs[tail_or_wing]["total_thickness_m"] - sandwich_specs[tail_or_wing]["core_thickness_m"]
    skin_cross_sectional_cirfumferance = cross_section_perimeter_length
    skin_cross_sectional_area = skin_cross_sectional_cirfumferance * skin_thickness
    total_skin_volume = span * skin_cross_sectional_area
    skin_density = assumed_and_set["gfrp_density_kgm3"]
    total_skin_mass = total_skin_volume * skin_density
    
    core_thickness = sandwich_specs[tail_or_wing]["core_thickness_m"]
    core_cross_sectional_cirfumferance = cross_section_perimeter_length
    core_cross_sectional_area = core_cross_sectional_cirfumferance * core_thickness
    total_core_volume = span * core_cross_sectional_area
    core_density = assumed_and_set["core_density_kgm3"]
    total_core_mass = total_core_volume * core_density

    print(f"skin mass: {total_skin_mass}")
    print(f"core mass: {total_core_mass}")
    
    total_wing_mass = total_skin_mass + total_core_mass

    return total_wing_mass

def get_wing_or_tail_mass(surface_area, assumed_and_set, tail_or_wing):

    k_wet = 2.1
    core_factor = 0.75 # part of the wing that actually has a core requirement

    skin_thickness = sandwich_specs[tail_or_wing]["total_thickness_m"] - sandwich_specs[tail_or_wing]["core_thickness_m"]
    m_faces = k_wet * surface_area * skin_thickness * assumed_and_set["gfrp_density_kgm3"]
    m_core = k_wet * core_factor * surface_area * sandwich_specs[tail_or_wing]["core_thickness_m"] * assumed_and_set["core_density_kgm3"]
    
    return m_core + m_faces  




def get_coefficients_at_ht_deflection(ht_deflection_angle, effective_angle_of_attack, dict_of_deflection_frames, phase):
    deflection_angles_array = np.array(range(-15,31,1))

    start_delta_e = deflection_angles_array[deflection_angles_array <= ht_deflection_angle].max()
    start_delta_e_frame = get_coefficients_at_alpha(dict_of_deflection_frames[f"{phase}_{start_delta_e}"], effective_angle_of_attack)
    start_delta_e_frame["alpha"] = start_delta_e
    end_delta_e = deflection_angles_array[deflection_angles_array >= ht_deflection_angle].min()
    end_delta_e_frame = get_coefficients_at_alpha(dict_of_deflection_frames[f"{phase}_{end_delta_e}"], effective_angle_of_attack)
    end_delta_e_frame["alpha"] = end_delta_e
    delta_e_frame = pd.DataFrame([start_delta_e_frame, end_delta_e_frame])

    return get_coefficients_at_alpha(delta_e_frame, ht_deflection_angle)


def get_coefficients_at_alpha(df, alpha):
    """
    Interpolate aerodynamic coefficients from XFOIL polar DataFrame.

    Parameters:
        df (pd.DataFrame): Must contain 'alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'
        alpha (float): Angle of attack in degrees

    Returns:
        dict: Interpolated coefficients at the specified alpha
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Ensure the DataFrame is sorted by alpha
    df_sorted = df.sort_values("alpha")

    # Check alpha bounds
    if not (df_sorted["alpha"].min() <= alpha <= df_sorted["alpha"].max()):
        raise ValueError(f"Alpha {alpha} is out of bounds ({df_sorted['alpha'].min()} to {df_sorted['alpha'].max()})")

    # Interpolate each column
    coeffs = {}
    for col in ["CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]:
        coeffs[col] = np.interp(alpha, df_sorted["alpha"], df_sorted[col])

    coeffs["alpha"] = alpha
    return coeffs

def get_cl_alpha_at(df, alpha, target_coeff, delta=0.5):
    """
    Approximates the local lift curve slope dCl/dα at a specified angle of attack.

    Parameters:
        df (pd.DataFrame): Polar data with columns 'alpha' and 'CL'
        alpha (float): Angle of attack in degrees at which to compute the slope
        delta (float): Small delta around alpha for finite difference (in degrees)

    Returns:
        float: Estimated dCl/dα (1/deg)
    """
    df = df.sort_values("alpha")

    alpha_min = -15
    alpha_max = 25

    if not (alpha_min <= alpha - delta and alpha + delta <= alpha_max):
        raise ValueError(f"Alpha ± delta must be within data bounds ({alpha_min} to {alpha_max})")

    # Interpolate CL at alpha ± delta
    target_coeff_plus = np.interp(alpha + delta, df["alpha"], df[target_coeff])
    target_coeff_minus = np.interp(alpha - delta, df["alpha"], df[target_coeff])

    target_coeff_alpha = (target_coeff_plus - target_coeff_minus) / (2 * delta)
    return target_coeff_alpha

def get_row_for_cl(df, target_cl):
    """
    Interpolate the polar data to find the row corresponding to a target CL.

    Parameters:
        df (pd.DataFrame): Must contain columns 'alpha' and 'CL'
        target_cl (float): Desired lift coefficient

    Returns:
        dict: Interpolated row with keys: alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr
    """
    df = df.sort_values("CL")

    if not (df["CL"].min() <= target_cl <= df["CL"].max()):

        raise ValueError(f"CL = {target_cl} is out of bounds ({df['CL'].min()} to {df['CL'].max()})")

    # Find bracketing rows
    lower_idx = df[df["CL"] <= target_cl]["CL"].idxmax()
    upper_idx = df[df["CL"] >= target_cl]["CL"].idxmin()

    row_low = df.loc[lower_idx]
    row_high = df.loc[upper_idx]

    if lower_idx == upper_idx:
        return row_low.to_dict()  # Exact match

    # Linear interpolation factor
    t = (target_cl - row_low["CL"]) / (row_high["CL"] - row_low["CL"])

    interpolated = {}
    for col in ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]:
        interpolated[col] = (1 - t) * row_low[col] + t * row_high[col]

    return interpolated


def initial_design_feasibility_pass(current_values, assumed_and_set):
    # Unpack Inputs
    mtow = current_values["mtow"]
    cl_cruise = current_values["cl_cruise"]
    cruise_speed_kmh = current_values["cruise_speed_kmh"]
    altitude_m = assumed_and_set.get("cruise_altitude_m", 0)
    rho = get_air_density(altitude_m)
    v_ms = kmh_to_ms(cruise_speed_kmh)

    # 1. Calculate Initial Wing Area
    current_values["wing_area_m2"] = (2 * mtow * g) / (rho * v_ms**2 * cl_cruise)

    # 2. Wing Loading
    current_values["wing_loading_pa"] = (mtow * g) / current_values["wing_area_m2"]
    

    # 3. Wing Geometry
    AR = current_values["aspect_ratio"]
    current_values["wing_span_m"] = (current_values["wing_area_m2"] * AR)**0.5
    current_values["chord_m"] = current_values["wing_area_m2"] / current_values["wing_span_m"]

    # 4. Fuselage Geometry
    fuselage_length = assumed_and_set["fuselage_length_mac_coeff"] * current_values["chord_m"]
    current_values["fuselage_body_length_m"] = fuselage_length
    current_values["wing_le_position_m"] = assumed_and_set["wing_le_position_fuselage_length_coeff"] * fuselage_length
    
    # 5. Tail Arm and Tail Areas
    tail_arm = assumed_and_set["horizontal_tail_arm_mac_coeff"] * current_values["chord_m"]
    current_values["tail_arm_m"] = tail_arm
    Vh = assumed_and_set["horizontal_tail_volume_coefficient"]
    Vv = assumed_and_set["vertical_tail_volume_coefficient"]
    current_values["horizontal_tail_area_m2"] = Vh * current_values["wing_area_m2"] * current_values["chord_m"] / tail_arm
    current_values["vertical_tail_area_m2"] = Vv * current_values["wing_area_m2"] * current_values["wing_span_m"] / tail_arm

    return current_values


def get_cm_thrust(current_values, hard_constraints, phase):

    phase_for_cg = ""

    if ("cruise" in phase) or (phase == "loiter"):
        phase_for_cg = phase
        alt = hard_constraints["cruise_altitude_m"]
        v_ms = kmh_to_ms(current_values[f"{phase_for_cg}_speed_kmh"])
        power_kw = current_values[f"{phase_for_cg}_power_kw"] 
    else:
        phase_for_cg = "cruiseout"
        alt = 0
        v_ms = 1.2 * kmh_to_ms(hard_constraints[f"stall_speed_kmh"])
        power_kw = engine_specs.get("max_continuous_power_kw")

    rho = get_air_density(alt)
    
    q = 0.5 * rho * v_ms**2

    # 4. Tail area (needed to convert moment to tail lift)
    S = current_values["wing_area_m2"]
    thrust = power_kw * 1000 / v_ms  # in N
    moment_thrust = - thrust * (0.5*current_values["fuselage_body_height_m"] - current_values[f"{phase_for_cg}_cg_from_floor_m"])  # in Nm

    cm_thrust = moment_thrust / (rho * v_ms**2 * S * current_values["chord_m"])

    return cm_thrust

def get_delt_cl_from_thrust(current_values, hard_constraints, phase):
    

    rho = get_air_density(hard_constraints["cruise_altitude_m"])
    v_ms = kmh_to_ms(current_values[f"{phase}_speed_kmh"])
    q = 0.5 * rho * v_ms**2

    # 4. Tail area (needed to convert moment to tail lift)
    S_h = current_values["horizontal_tail_area_m2"]
    thrust = current_values[f"{phase}_power_kw"] * 1000 / v_ms  # in N
    moment_thrust = - thrust * current_values[f"{phase}_cg_from_floor_m"]  # in Nm
    delta_cl_tail_thrust = moment_thrust / (q * S_h * current_values["tail_arm_m"])

    return delta_cl_tail_thrust

def tail_geometry(area_m2, aspect_ratio):
    span_m = (area_m2 * aspect_ratio) ** 0.5
    chord_m = area_m2 / span_m
    return span_m, chord_m


def _stability_analysis(
    chord_m,
    cg_m,
    tail_volume_coeff,
    wing_le_position_m
):
    Vh = tail_volume_coeff

    neutral_point_m = wing_le_position_m + 0.25 * chord_m + Vh * chord_m
    static_margin = (neutral_point_m - cg_m) / chord_m


    return {
        "neutral_point_m": neutral_point_m,
        "static_margin": static_margin,
    }

def stability_analysis(
    current_values,
    assumed_and_set,
    deflections_dict,
    phase
):

    phase_for_delta = ""
    if ("cruise" in phase) or (phase == "loiter"):
        phase_for_delta = "cruise"
    elif phase in ["takeoff", "landing"]:
        phase_for_delta = "takeoff"

    # cl_row = get_row_for_cl(deflections_dict[f"{phase_for_delta}_0"], current_values[f"{phase}_cl"])

    wing_ac = current_values["wing_le_position_m"] + 0.25*current_values["chord_m"]

    coeff_ratio = get_cl_alpha_at(deflections_dict[f"{phase_for_delta}_0"], current_values[f'{phase}_angle_of_attack_deg']  + assumed_and_set["wing_incident_angle"], "CM", 0.1) / get_cl_alpha_at(deflections_dict[f"{phase_for_delta}_0"], current_values[f'{phase}_angle_of_attack_deg']  + assumed_and_set["wing_incident_angle"], "CL", 0.1)
    # new_wing_ac = wing_ac - (cl_row['CM']/cl_row['CL']) * current_values["chord_m"]
    new_wing_ac = wing_ac - coeff_ratio * current_values["chord_m"]

    eta_h = calculate_eta_h(current_values, phase=phase)

    # neutral_point_m = current_values[f"{phase}_x_ac_w_m"] + assumed_and_set["horizontal_tail_volume_coefficient"] * current_values['chord_m']
    neutral_point_m = new_wing_ac + (eta_h * assumed_and_set["horizontal_tail_volume_coefficient"] * current_values["chord_m"] * (1-assumed_and_set["ht_downwash_efficiency_coeff"]))
    static_margin = (neutral_point_m - current_values[f"{phase}_cg_from_nose_m"]) / current_values['chord_m']

    # x_cg = current_values[f"{phase}_cg_from_nose_m"]
    # h_np = (neutral_point_m - current_values["wing_le_position_m"]) / current_values["chord_m"]
    # h_cg = (x_cg - current_values["wing_le_position_m"]) / current_values["chord_m"]
    # static_margin = h_np - h_cg

    return {
        "neutral_point_m": neutral_point_m,
        "static_margin": static_margin,
    }


def _interp_col_vs_alpha(df, alpha_deg, col):
    # df must have columns 'alpha' and the requested col
    d = df.sort_values("alpha")
    return float(np.interp(alpha_deg, d["alpha"].values, d[col].values))

def CL_tail_bilinear(alpha_t_deg, delta_e_deg, deflections_dict, phase="cruise"):
    """CL_t at (alpha_t, delta_e) by blending between two δe polars."""
    # available deflection tables like 'cruise_-10', 'cruise_0', 'cruise_10', ...
    defs = sorted(int(k.split("_")[1]) for k in deflections_dict if k.startswith(f"{phase}_"))
    # bracket delta_e
    j = np.searchsorted(defs, delta_e_deg)
    
    j0 = max(0, min(j-1, len(defs)-2))
    j1 = j0 + 1
    d0, d1 = defs[j0], defs[j1]
    w = 0.0 if d1 == d0 else (delta_e_deg - d0) / (d1 - d0)

    df0 = deflections_dict[f"{phase}_{d0}"]
    df1 = deflections_dict[f"{phase}_{d1}"]

    CL0 = _interp_col_vs_alpha(df0, alpha_t_deg, "CL")
    CL1 = _interp_col_vs_alpha(df1, alpha_t_deg, "CL")
    return (1.0 - w)*CL0 + w*CL1

def solve_delta_e_for_CLt(CLt_req, alpha_t_deg, deflections_dict, phase="cruise",
                          dmin=None, dmax=None, tol=1e-4, itmax=40):
    """Robust 1-D solve for δe: CL_tail(alpha_t, δe) = CLt_req."""
    defs = sorted(int(k.split("_")[1]) for k in deflections_dict if k.startswith(f"{phase}_"))
    if dmin is None: dmin = defs[0]
    if dmax is None: dmax = defs[-1]

    # bracket
    f_lo = CL_tail_bilinear(alpha_t_deg, dmin, deflections_dict, phase) - CLt_req
    f_hi = CL_tail_bilinear(alpha_t_deg, dmax, deflections_dict, phase) - CLt_req

    # if not bracketed (rare), nudge toward center
    if f_lo*f_hi > 0:
        x0 = 0.5*(dmin + dmax) - 5.0
        x1 = 0.5*(dmin + dmax) + 5.0
        f0 = CL_tail_bilinear(alpha_t_deg, x0, deflections_dict, phase) - CLt_req
        for _ in range(itmax):
            f1 = CL_tail_bilinear(alpha_t_deg, x1, deflections_dict, phase) - CLt_req
            if abs(f1) < tol: return float(np.clip(x1, dmin, dmax))
            denom = (f1 - f0) if abs(f1 - f0) > 1e-9 else 1e-9
            x2 = x1 - f1*(x1 - x0)/denom
            x0, f0, x1 = x1, f1, float(np.clip(x2, dmin, dmax))
        return float(np.clip(x1, dmin, dmax))

    # bisection
    lo, hi = float(dmin), float(dmax)
    for _ in range(itmax):
        mid = 0.5*(lo + hi)
        f_mid = CL_tail_bilinear(alpha_t_deg, mid, deflections_dict, phase) - CLt_req
        if abs(f_mid) < tol: return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5*(lo + hi)


def closed_form_trim_analysis(current_values, assumed_and_set, hard_constraints, deflections_dict, phase):
    """
    Computes wing CL, tail CL, wing/tail AoA and required elevator deflection
    for a given flight phase using real moment equilibrium (uses Cm_wing).
    """
    # --- Normalise phase naming ---
    if ("cruise" in phase) or (phase == "loiter"):
        phase_for_delta = "cruise"
        rho = get_air_density(hard_constraints["cruise_altitude_m"])
    elif phase in ["takeoff", "landing"]:
        phase_for_delta = "takeoff"
        rho = get_air_density(0.0)
    else:
        phase_for_delta = "cruise"
        rho = get_air_density(hard_constraints["cruise_altitude_m"])

    # --- Pull values from dicts ---
    # Use per-phase mass if you have it; otherwise this falls back to mtow (not ideal)
    m   = current_values.get(f"{phase}_mass_kg", current_values["mtow"])
    V   = kmh_to_ms(current_values[f"{phase}_speed_kmh"])
    Sw  = current_values["wing_area_m2"]
    Sh  = current_values["horizontal_tail_area_m2"]
    c   = current_values["chord_m"]
    lt  = current_values["tail_arm_m"]
    q   = 0.5 * rho * V**2
    W   = m * 9.81

    # Incidences and downwash
    i_w   = float(assumed_and_set.get("wing_incident_angle", 0.0))
    i_h   = float(assumed_and_set.get("ht_incident_angle",   0.0))
    deda  = float(assumed_and_set.get("ht_downwash_efficiency_coeff", 0.3))  # dε/dα
    eta_h = current_values.get(f"{phase}_tail_dynamic_pressure_ratio", 1.0)           # q_t / q

    # --- First pass: wing-only CL to get the wing operating point ---
    CL_w = W / (q * Sw)   # initial guess (tail force not yet removed)

    # Wing AC x-location (m from nose); prefer phase-specific if available
    x_ac_w = current_values.get(f"{phase}_x_ac_w_m",
                                current_values["wing_le_position_m"] + 0.25 * c)
    x_cg   = current_values[f"{phase}_cg_from_nose_m"]

    # --- Wing AoA & Cm_wing from polar at this CL_w ---
    row_w   = get_row_for_cl(deflections_dict[f"{phase_for_delta}_0"], CL_w)
    alpha_w = float(row_w["alpha"]) - i_w       # body AoA (deg)
    Cm_w    = float(row_w["CM"])                # THIS is the Cm_wing you asked about

    # Tail local α (deg) after downwash plus tail incidence
    alpha_t = (1.0 - deda) * alpha_w + i_h

    # --- Moment equilibrium about CG: solve for CL_t (THIS is where Cm_wing enters) ---
    d_w   = (x_ac_w - x_cg) / c
    denom = eta_h * (Sh/Sw) * (lt/c)
    if abs(denom) < 1e-9:
        raise ValueError("Tail moment denominator nearly zero (check lt, Sh, eta_h).")

    Cm_fuse   = 0.0                 # add if you have it
    Cm_thrust = get_cm_thrust(current_values, hard_constraints, phase)                 # add if you model thrust pitching moment

    # 

    CL_t = - (Cm_w + CL_w * d_w + Cm_fuse + Cm_thrust) / denom

    # print(f"Cm_wing: {Cm_w:.4f}, Cm_fuse: {Cm_fuse:.4f}, Cm_thrust: {Cm_thrust:.4f}, CL_t: {CL_t:.4f}, denom: {denom: .4f}")
    # Optional: refine CL_w by removing the tail lift from force balance (one correction step)
    L_tail = (eta_h * q) * Sh * CL_t
    CL_w = (W - L_tail) / (q * Sw)

    # Re-read wing polar and Cm if you want a second, tighter pass:
    row_w   = get_row_for_cl(deflections_dict[f"{phase_for_delta}_0"], CL_w)
    alpha_w = float(row_w["alpha"]) - i_w
    Cm_w    = float(row_w["CM"])
    alpha_t = (1.0 - deda) * alpha_w + i_h
    

    # --- Elevator deflection to achieve CL_t at alpha_t ---
    delta_e = solve_delta_e_for_CLt(CL_t, alpha_t, deflections_dict, phase=phase_for_delta)

    return {
        "cl_wing": CL_w,
        "cl_tail_required": CL_t,
        "alpha_wing_deg": alpha_w,   # body AoA
        "alpha_tail_deg": alpha_t,   # tail local geometric AoA
        "delta_elevator_deg": delta_e,
    }



def effective_CLmax_partial_span(
    deflections_dict,
    assumed_and_set,
    phase_prefix="takeoff",
    alpha_margin_deg=0.0              # optional: back off a little from the formal max if desired
):
    """
    Returns:
      {
        "CLmax_eff": float,
        "alpha_at_CLmax_deg": float,
        "lambda_eff": float
      }

    Method:
      1) Build a common α grid over the overlap of the clean and flapped tables.
      2) For each α, blend CL(α):  CL_eff(α) = (1-λ) CL_clean(α) + λ CL_flap(α),
         where λ = spanwise_flap_effectiveness * flap_span_fraction, clipped to [0,1].
      3) Take the maximum CL_eff over that α grid. Optionally subtract a safety margin.
    """
    import numpy as np

    flap_deflection_deg = assumed_and_set["flap_deflection_deg"]
    flap_span_fraction = assumed_and_set["flap_span_fraction"]
    spanwise_flap_effectiveness = assumed_and_set["spanwise_flap_effectiveness"]

    lam = max(0.0, min(1.0, spanwise_flap_effectiveness * flap_span_fraction))

    # Pick keys and fallbacks
    base_key = f"{phase_prefix}_0" if f"{phase_prefix}_0" in deflections_dict else "cruise_0"
    flap_key = f"{phase_prefix}_{int(flap_deflection_deg)}"
    if flap_key not in deflections_dict:
        # No flapped table available → no change from clean
        CLmax = float(deflections_dict[base_key]["CL"].max())
        a_at  = float(deflections_dict[base_key].loc[deflections_dict[base_key]["CL"].idxmax()]["alpha"])
        return {"CLmax_eff": CLmax, "alpha_at_CLmax_deg": a_at, "lambda_eff": lam}

    df_clean = deflections_dict[base_key][["alpha","CL", "CD", "CM"]].dropna().sort_values("alpha")
    df_flap  = deflections_dict[flap_key][["alpha","CL", "CD", "CM"]].dropna().sort_values("alpha")

    # Build common α grid over the overlap
    a_lo = max(df_clean["alpha"].min(), df_flap["alpha"].min())
    a_hi = min(df_clean["alpha"].max(), df_flap["alpha"].max())
    if a_hi <= a_lo:
        # Degenerate overlap → fall back to clean
        CLmax = float(df_clean["CL"].max())
        a_at  = float(df_clean.loc[df_clean["CL"].idxmax()]["alpha"])
        CD_at_cl_max = float(df_clean.loc[df_clean["CL"].idxmax()]["CD"])
        CM_at_cl_max = float(df_clean.loc[df_clean["CL"].idxmax()]["CM"])
        return {"CLmax_eff": CLmax, "CD_at_CLmax_eff": CD_at_cl_max, "CM_at_CLmax_eff" : CM_at_cl_max, "alpha_at_CLmax_deg": a_at, "lambda_eff": lam}

    alpha_grid = np.linspace(a_lo, a_hi, 801)

    # Interpolate CL vs α for both tables
    CL_clean = np.interp(alpha_grid, df_clean["alpha"].values, df_clean["CL"].values)
    CL_flap  = np.interp(alpha_grid, df_flap["alpha"].values,  df_flap["CL"].values)

    CD_clean = np.interp(alpha_grid, df_clean["alpha"].values, df_clean["CD"].values)
    CD_flap  = np.interp(alpha_grid, df_flap["alpha"].values,  df_flap["CD"].values)

    CM_clean = np.interp(alpha_grid, df_clean["alpha"].values, df_clean["CM"].values)
    CM_flap  = np.interp(alpha_grid, df_flap["alpha"].values,  df_flap["CM"].values)

    # Blend at same α
    CL_eff = (1.0 - lam) * CL_clean + lam * CL_flap
    CD_eff = (1.0 - lam) * CD_clean + lam * CD_flap
    CM_eff = (1.0 - lam) * CM_clean + lam * CM_flap

    # Pick the maximum (optionally back off a little)
    idx = int(np.argmax(CL_eff))
    a_star = float(alpha_grid[idx])
    CL_star = float(CL_eff[idx])
    CD_star = float(CD_eff[idx])
    CM_star = float(CM_eff[idx])

    if alpha_margin_deg > 0.0:
        # Back off: find the CL at (a_star - margin) if within bounds
        a_safe = max(alpha_grid[0], a_star - alpha_margin_deg)
        CL_safe = float(np.interp(a_safe, alpha_grid, CL_eff))
        CD_safe = float(np.interp(a_safe, alpha_grid, CD_eff))
        CM_safe = float(np.interp(a_safe, alpha_grid, CM_eff))
        return {"CLmax_eff": CL_safe, "CD_at_CLmax_eff": CD_safe, "CM_at_CLmax_eff" : CM_safe,"alpha_at_CLmax_deg": a_safe, "lambda_eff": lam}

    return {"CLmax_eff": CL_star, "CD_at_CLmax_eff": CD_star, "CM_at_CLmax_eff" : CM_star, "alpha_at_CLmax_deg": a_star, "lambda_eff": lam}

def effective_CD0_partial_span(
    deflections_dict,
    assumed_and_set,
    phase_prefix="takeoff",
    alpha_bounds_deg=(-8.0, 18.0),
    tol=1e-5, max_iter=40
):
    """
    CD0 for partial-span flaps by blending clean/flapped polars at the SAME alpha,
    then solving CL_eff(alpha0)=0 and reading CD_eff(alpha0).
    Returns: {"CD0_eff": float, "alpha_at_CL0_deg": float, "lambda_eff": float}
    """

    flap_deflection_deg = assumed_and_set["flap_deflection_deg"]
    flap_span_fraction = assumed_and_set["flap_span_fraction"]
    spanwise_flap_effectiveness = assumed_and_set["spanwise_flap_effectiveness"]

    lam = max(0.0, min(1.0, spanwise_flap_effectiveness * flap_span_fraction))
    base_key = f"{phase_prefix}_0" if f"{phase_prefix}_0" in deflections_dict else "cruise_0"
    flap_key = f"{phase_prefix}_{int(flap_deflection_deg)}"

    if flap_key not in deflections_dict:
        row0 = get_row_for_cl(deflections_dict[base_key], 0.0)
        return {"CD0_eff": float(row0["CD"]), "alpha_at_CL0_deg": float(row0["alpha"]), "lambda_eff": 0.0}

    def coeffs_at_alpha(alpha_deg):
        rc = get_coefficients_at_alpha(deflections_dict[base_key], alpha_deg)
        rf = get_coefficients_at_alpha(deflections_dict[flap_key], alpha_deg)
        blend = lambda a, b: (1.0 - lam)*a + lam*b
        return {"CL": blend(rc["CL"], rf["CL"]), "CD": blend(rc["CD"], rf["CD"])}

    lo, hi = alpha_bounds_deg
    def f(a): return coeffs_at_alpha(a)["CL"]

    flo, fhi = f(lo), f(hi)
    guard = 0
    while flo * fhi > 0.0 and guard < 6:
        lo -= 2.0; hi += 2.0
        flo, fhi = f(lo), f(hi)
        guard += 1
    if flo * fhi > 0.0:
        row0 = get_row_for_cl(deflections_dict[base_key], 0.0)
        return {"CD0_eff": float(row0["CD"]), "alpha_at_CL0_deg": float(row0["alpha"]), "lambda_eff": lam}

    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        fm = f(mid)
        if abs(fm) < tol:
            cd0 = float(coeffs_at_alpha(mid)["CD"])
            return {"CD0_eff": cd0, "alpha_at_CL0_deg": mid, "lambda_eff": lam}
        if fm * flo <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm

    alpha0 = 0.5*(lo+hi)
    cd0 = float(coeffs_at_alpha(alpha0)["CD"])
    return {"CD0_eff": cd0, "alpha_at_CL0_deg": alpha0, "lambda_eff": lam}

def wing_area_from_stall_speed(
    cur, hard_constraints, deflections_dict, assumed_and_set,
    alpha_margin_deg=0.0, altitude_m=0.0
):
    """
    Sizing by Vs target with partial-span flaps.
    Uses your effective_CLmax_partial_span (α-aware).
    """

    rho = get_air_density(altitude_m)
    g = 9.81
    W = cur["mtow"] * g
    Vs_ms = kmh_to_ms(hard_constraints["stall_speed_kmh"])

    V_LOF = 1.2 * Vs_ms
    pack = effective_CLmax_partial_span(
        deflections_dict, assumed_and_set, phase_prefix="takeoff",
        alpha_margin_deg=alpha_margin_deg
    )
    CLmax_eff = pack["CLmax_eff"]

    Sw = W / (0.5 * rho * V_LOF**2 * CLmax_eff)
    return {"wing_area_m2": Sw, "CLmax_eff": CLmax_eff, "alpha_at_CLmax_deg": pack["alpha_at_CLmax_deg"]}


def wing_area_from_takeoff_distance(
    cur, assumed_and_set, hard_constraints,
    engine_specs, propeller_specs, deflections_dict,
    thrust_static_cap_factor=1.1,   # caps T ~ factor * (Pavail / V_LOF)
    altitude_m=0.0,
    max_iter=40, tol=0.5
):
    """
    Solve for wing area S so that computed ground roll ≤ hard_constraints['takeoff_distance_max_m'].

    Model:
      m * dV/dt = T(V) - D(V,S) - μ (W - L(V,S))
      dx/dt = V
      stop when V ≥ 1.2 * Vs_flap (LOF criterion)
    """

    cl_fraction_of_max = assumed_and_set["takeoff_cl_fraction_from_max"]
    oswald_e_TO = assumed_and_set["oswald_derated"]

    g = 9.81
    rho = get_air_density(altitude_m)
    W = cur["mtow"] * g
    m = W / g
    ARw = float(assumed_and_set.get("aspect_ratio", 12.0))
    mu_roll = float(assumed_and_set["rolling_resistance_coefficient"])

    # Flap-effective CLmax and CD0
    cl_pack = effective_CLmax_partial_span(
        deflections_dict, assumed_and_set, "takeoff", 0.0
    )
    cd0_pack = effective_CD0_partial_span(
        deflections_dict, assumed_and_set, "takeoff"
    )

    CLmax_eff = cl_pack["CLmax_eff"]
    CD0_eff = cd0_pack["CD0_eff"]

    # Induced factor
    k_ind = 1.0 / (math.pi * ARw * oswald_e_TO)

    # Power available (takeoff)
    eta_prop_TO = propeller_specs["efficiency"].get("takeoff", propeller_specs["efficiency"].get("take-off", 0.75))
    eta_gb = float(engine_specs.get("gear_box_efficiency", 1.0))
    P_avail_W = float(engine_specs["max_power_kw"]) * 1000.0 * eta_prop_TO * eta_gb

    s_target = float(hard_constraints["takeoff_distance_max_m"])

    def ground_roll_for_S(Sw):
        # determine Vs with flaps for this S
        Vs = math.sqrt(W / (0.5 * rho * Sw * CLmax_eff))
        V_LOF = 1.2 * Vs

        # cap thrust near static using V_LOF
        T_cap = thrust_static_cap_factor * (P_avail_W / max(V_LOF, 1.0))

        # average CL during roll
        CL_avg = cl_fraction_of_max * CLmax_eff

        # integrate with fixed dt in speed domain (simple, robust)
        V = 0.1  # m/s to avoid div by zero
        x = 0.0
        dt = 0.05
        while V < V_LOF and x < 5 * s_target:
            q = 0.5 * rho * V**2
            L = q * Sw * CL_avg
            CD = CD0_eff + k_ind * (CL_avg**2)
            D = q * Sw * CD
            T = min(P_avail_W / max(V, 1.0), T_cap)
            a = (T - D - mu_roll * (W - L)) / m
            a = max(a, 0.0)
            V += a * dt
            x += V * dt
        return x, V_LOF

    # Solve for S by simple bracket + secant on f(S)=x(S)-s_target
    # Bracket
    S_lo = max(0.5, W / (0.5 * rho * kmh_to_ms(hard_constraints["stall_speed_kmh"])**2 * CLmax_eff)) * 0.5
    S_hi = S_lo * 4.0
    x_lo, _ = ground_roll_for_S(S_lo)
    x_hi, _ = ground_roll_for_S(S_hi)

    guard = 0
    while (x_lo > s_target) and guard < 12:
        S_lo *= 1.2
        x_lo, _ = ground_roll_for_S(S_lo)
        guard += 1
    guard = 0
    while (x_hi < s_target) and guard < 12:
        S_hi *= 1.5
        x_hi, _ = ground_roll_for_S(S_hi)
        guard += 1

    S0, S1 = S_lo, S_hi
    x0, x1 = x_lo, x_hi

    for _ in range(max_iter):
        if abs(x1 - x0) < 1e-6:
            S_try = 0.5*(S0 + S1)
        else:
            # secant
            S_try = S1 + (s_target - x1) * (S1 - S0) / (x1 - x0)

        S_try = max(S_lo*0.5, min(S_hi*2.0, S_try))
        x_try, V_LOF = ground_roll_for_S(S_try)
        if abs(x_try - s_target) < tol:
            return {
                "wing_area_m2": float(S_try),
                "ground_roll_m": float(x_try),
                "V_LOF_ms": float(V_LOF),
                "CLmax_eff": CLmax_eff,
                "CD0_eff": CD0_eff
            }
        S0, x0 = S1, x1
        S1, x1 = S_try, x_try

    return {
        "wing_area_m2": float(S1),
        "ground_roll_m": float(x1),
        "V_LOF_ms": float(V_LOF),
        "CLmax_eff": CLmax_eff,
        "CD0_eff": CD0_eff
    }

def climb_rate_ms(
    cur, assumed_and_set, engine_specs, propeller_specs, deflections_dict,
    wing_area_m2,
    altitude_m, speed_kmh,
    config="clean",  # "takeoff" or "clean"
    flap_deflection_deg=0, flap_span_fraction=0.50, spanwise_flap_effectiveness=0.95,
    oswald_e=0.85
):
    """
    ROC at (altitude, speed). Uses partial-span CD0 (config-dependent).
    """
    import math
    rho = get_air_density(altitude_m)
    V = kmh_to_ms(speed_kmh)
    g = 9.81
    W = cur.get("climb_mass_kg", cur.get("takeoff_mass_kg", cur["mtow"])) * g
    ARw = float(assumed_and_set.get("aspect_ratio", 12.0))
    k_ind = 1.0 / (math.pi * ARw * oswald_e)

    # CD0
    if config == "takeoff" and flap_deflection_deg != 0:
        cd0_pack = effective_CD0_partial_span(
            deflections_dict, assumed_and_set, phase_prefix="takeoff",
        )
        CD0 = cd0_pack["CD0_eff"]
    else:
        # clean CD0 at CL=0 from clean polar
        row0 = get_row_for_cl(deflections_dict.get("cruise_0", deflections_dict["takeoff_0"]), 0.0)
        CD0 = float(row0["CD"])

    q = 0.5 * rho * V**2
    CL = W / (q * wing_area_m2)
    CD = CD0 + k_ind * CL**2
    D = q * wing_area_m2 * CD

    # Power available at this condition (simple cap by engine max power and prop η)
    eta_gb = float(engine_specs.get("gear_box_efficiency", 1.0))
    # You can curve-fit η_prop(V,alt). Here: use "climb" if present, else "cruise".
    eta_prop = propeller_specs["efficiency"].get("climb",
                  propeller_specs["efficiency"].get("cruise", 0.80))
    P_max_W = float(engine_specs.get("max_power_kw", engine_specs.get("max_cruise_power_kw", 0.0))) * 1000.0
    P_av = eta_prop * eta_gb * P_max_W

    P_req = D * V
    ROC = max(0.0, (P_av - P_req) / W)
    return ROC

def time_to_altitude(
    cur, assumed_and_set, engine_specs, propeller_specs, deflections_dict,
    wing_area_m2,
    alt_start_m, alt_end_m,
    speed_mode="best",   # "best" or "given"
    speed_kmh_given=None,
    config="clean",
    flap_deflection_deg=0, flap_span_fraction=0.50, spanwise_flap_effectiveness=0.95,
    oswald_e=0.85,
    steps=40
):
    """
    Integrate dt = dh / ROC(h). If speed_mode="best", maximize ROC over a speed grid
    per layer. Returns hours and an ROC profile list.
    """
    import numpy as np

    h_grid = np.linspace(alt_start_m, alt_end_m, steps+1)
    total_time_s = 0.0
    roc_profile = []

    for i in range(steps):
        h_lo, h_hi = float(h_grid[i]), float(h_grid[i+1])
        h_mid = 0.5*(h_lo + h_hi)
        dh = h_hi - h_lo

        if speed_mode == "best":
            # grid around 1.3 Vs (clean or flap)
            rho_mid = get_air_density(h_mid)
            g = 9.81
            W = cur.get("climb_mass_kg", cur.get("takeoff_mass_kg", cur["mtow"])) * g

            if config == "takeoff" and flap_deflection_deg != 0:
                cl_pack = effective_CLmax_partial_span(
                    deflections_dict, assumed_and_set, phase_prefix="takeoff",
                )
                CLmax_eff = cl_pack["CLmax_eff"]
            else:
                # clean CLmax from clean table for Vs estimate
                CLmax_eff = float(deflections_dict.get("cruise_0", deflections_dict["takeoff_0"])["CL"].max())

            Vs_mid = np.sqrt(W / (0.5 * rho_mid * wing_area_m2 * CLmax_eff))
            speeds = np.linspace(1.15*Vs_mid, 1.6*Vs_mid, 9)  # m/s
            best_roc = 0.0
            best_kmh = None
            for V in speeds:
                roc = climb_rate_ms(
                    cur, assumed_and_set, engine_specs, propeller_specs, deflections_dict,
                    wing_area_m2, h_mid, V*3.6,
                    config=config, flap_deflection_deg=flap_deflection_deg,
                    flap_span_fraction=flap_span_fraction,
                    spanwise_flap_effectiveness=spanwise_flap_effectiveness,
                    oswald_e=oswald_e
                )
                if roc > best_roc:
                    best_roc = roc; best_kmh = V*3.6
            ROC_mid = best_roc
            V_used_kmh = best_kmh
        else:
            if speed_kmh_given is None:
                raise ValueError("Provide speed_kmh_given when speed_mode='given'.")
            ROC_mid = climb_rate_ms(
                cur, assumed_and_set, engine_specs, propeller_specs, deflections_dict,
                wing_area_m2, h_mid, speed_kmh_given,
                config=config, flap_deflection_deg=flap_deflection_deg,
                flap_span_fraction=flap_span_fraction,
                spanwise_flap_effectiveness=spanwise_flap_effectiveness,
                oswald_e=oswald_e
            )
            V_used_kmh = speed_kmh_given

        ROC_mid = max(ROC_mid, 1e-3)  # avoid divide by zero
        dt = dh / ROC_mid  # seconds
        total_time_s += dt
        roc_profile.append({"h_mid_m": h_mid, "ROC_ms": ROC_mid, "V_used_kmh": V_used_kmh})

    return {"time_mins": total_time_s/60.0,"profile": roc_profile}

def update_time_to_altitude_and_ROC(predrop, assumed_and_set, hard_constraints, engine_specs, propeller_specs, deflections_dict):

    time_to_alt_before_retraction = time_to_altitude(
                predrop, assumed_and_set, engine_specs, propeller_specs, deflections_dict,
                wing_area_m2 = predrop["wing_area_m2"],
                alt_start_m = 0, alt_end_m = 200,
                speed_mode="best",   # "best" or "given"
                speed_kmh_given=None,
                config="takeoff",
                flap_deflection_deg=20, flap_span_fraction=0.50, spanwise_flap_effectiveness=0.95,
                oswald_e=assumed_and_set['oswald_derated'],
                steps=16
            )

    time_to_alt_after_retraction = time_to_altitude(
                predrop, assumed_and_set, engine_specs, propeller_specs, deflections_dict,
                wing_area_m2 = predrop["wing_area_m2"],
                alt_start_m = 200, alt_end_m = hard_constraints["cruise_altitude_m"],
                speed_mode="best",   # "best" or "given"
                speed_kmh_given=None,
                config="clean",
                flap_deflection_deg=0, flap_span_fraction=0.50, spanwise_flap_effectiveness=0.95,
                oswald_e=assumed_and_set['oswald_clean'],
                steps=16
            )

    time_to_alt_total = time_to_alt_before_retraction['time_mins'] + time_to_alt_after_retraction['time_mins'] 
    predrop['climb_time_to_altitude_min'] = time_to_alt_total

    climb_profile = time_to_alt_before_retraction['profile'] + time_to_alt_before_retraction['profile']
    average_roc = sum([step["ROC_ms"] for step in climb_profile]) / len(climb_profile)
    predrop['average_climb_rate_mps'] = average_roc

    return predrop