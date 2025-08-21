from scipy.interpolate import interp1d
import math
import numpy as np
from ADRpy import atmospheres as at
import ussa1976
from specs import *

# functions

def kmh_to_ms(v_kmh):
    return v_kmh / 3.6

def ms_to_kmh(v_ms):
    return v_ms * 3.6

def calc_required_wing_area(mtow_kg, cl_max, rho, v_stall_ms):
    weight_n = mtow_kg * 9.81
    s = (2 * weight_n) / (rho * v_stall_ms**2 * cl_max)
    return s

def calc_stall_speed(mtow_kg, wing_area, cl_max, rho):
    weight_n = mtow_kg * 9.81
    v_stall = math.sqrt((2 * weight_n) / (rho * wing_area * cl_max))
    return v_stall

def calc_takeoff_distance(v_to_ms, acceleration):
    return v_to_ms**2 / (2 * acceleration)

def calc_acceleration(thrust_n, mass_kg, mu, weight_n):
    friction = mu * weight_n
    net_force = thrust_n - friction
    return net_force / mass_kg

def calc_fuel_needed(power_kw, sfc, hours):
    return power_kw * sfc * hours

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


# Constants
g = 9.81  # m/s²
atm = at.Atmosphere()

# Helper: convert km/h to m/s
def kmh_to_ms(v_kmh):
    return v_kmh / 3.6

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


# --- 2. Calculate Lift Coefficient required for steady cruise ---
def calc_cl_cruise(mtow_kg, v_kmh, wing_area_m2, altitude_m):
    rho = get_air_density(altitude_m)
    v = kmh_to_ms(v_kmh)
    cl = (mtow_kg * g) / (0.5 * rho * v**2 * wing_area_m2)
    return cl

# --- 3. Estimate Drag Coefficient using parabolic drag polar ---
def calc_cd_total(cd0, cl, aspect_ratio, e=0.8):
    k = 1 / (np.pi * e * aspect_ratio)
    return cd0 + k * cl**2

# --- 4. Compute Required Power for Cruise ---
def calc_power_required(mtow_kg, v_kmh, wing_area_m2, cd0, aspect_ratio, altitude_m, prop_eff):
    rho = get_air_density(altitude_m)
    v = kmh_to_ms(v_kmh)
    cl = calc_cl_cruise(mtow_kg, v_kmh, wing_area_m2, altitude_m)
    cd = calc_cd_total(cd0, cl, aspect_ratio)
    drag = 0.5 * rho * v**2 * wing_area_m2 * cd
    power_required = drag * v / prop_eff  # in watts
    return power_required / 1000  # in kW

# --- 5. Compute Cruise Fuel Burn ---
def calc_cruise_fuel(mtow_kg, v_kmh, wing_area_m2, cd0, aspect_ratio, altitude_m, prop_eff, cruise_time_hr, cruise_power_kw):
    sfc_data = estimate_cruise_sfc_from_dicts(engine_power_to_rpm, engine_sfc_to_rpm, cruise_power_kw)
    sfc_kgperkwh = sfc_data["sfc_cruise"]
    fuel_burn = cruise_power_kw * cruise_time_hr * sfc_kgperkwh
    return fuel_burn, sfc_data["rpm_cruise"], sfc_kgperkwh

def calculate_cg(weights_dict_input, positions_dict):

    weights_dict = weights_dict_input.copy()

    del weights_dict["horizontal_tail"]
    del weights_dict["vertical_tail"]
    
    numerator = sum(weights_dict[k] * positions_dict[k] for k in weights_dict)
    denominator = sum(weights_dict.values())
    return numerator / denominator

def _estimate_component_positions(fuselage_length, wing_le_position, chord, tail_arm, internal_payload_length, fuselage_fuel_tank_length):
    if cg_estimate is None:
        cg_estimate = wing_le_position + 0.25 * chord  # Initial guess at 25% MAC

    return {
        "fuselage": 0.5 * fuselage_length,
        "wing": wing_le_position + 0.45 * chord,
        # "tails": tail_arm + wing_le_position + 0.25 * chord,
        "tails": 0.9*fuselage_length,
        "engine": fuselage_length * 0.95,
        "propeller": fuselage_length * 1.0,
        "internal_payload": fuselage_length - 1 - (0.5*internal_payload_length), # 1 is the fuselage length allotted for the engine and its accessories
        "wing_payload": wing_le_position + 0.5 * chord,
        # "payload": cg_estimate,
        # "wing_fuel": wing_le_position + 0.5 * chord,
        "wing_fuel": wing_le_position + 0.45 * chord,
        "fuselage_fuel": wing_le_position - (0.5*fuselage_fuel_tank_length),
        "avionics": fuselage_length * 0.1,
        "landing_gear": cg_estimate - 0.1 * fuselage_length,
        "misc": cg_estimate
    }

def estimate_component_positions(current_values, hard_constraints, assumed_and_set, weights_dict_kg_no_fuel):

    cg_estimate = current_values[f"cruiseout_cg_from_nose_m"]

    tail_local_cg_ht_z = weights_dict_kg_no_fuel["horizontal_tail"] * (current_values["h_tail_chord_m"] * assumed_and_set["wing_airfoil_thickness_ratio"]) / 2
    tail_local_cg_vt_z = weights_dict_kg_no_fuel["vertical_tail"] * current_values["v_tail_span_m"] / 2
    tail_local_cg_z = (tail_local_cg_ht_z + tail_local_cg_vt_z) / weights_dict_kg_no_fuel["tails"]

    full_dict = {
        "fuselage": (0.5 * current_values["fuselage_body_length_m"], 0.5*current_values["fuselage_body_height_m"]),
        "wing": (current_values["wing_le_position_m"] + 0.45 * current_values["chord_m"], assumed_and_set["wing_airfoil_thickness_ratio"] * current_values["chord_m"]),
        "tails": (current_values["x_ht_le_m"] + assumed_and_set["tail_mass_cg_from_le_coeff"]  * current_values["h_tail_chord_m"], tail_local_cg_z + current_values["tail_boom_pylon_height_m"]),
        # "tails": 0.9*fuselage_length,
        "engine": (current_values["fuselage_body_length_m"] * 0.0, 0.5*current_values["fuselage_body_height_m"]),
        "propeller": (current_values["fuselage_body_length_m"] * -0.05, 0.5*current_values["fuselage_body_height_m"] ),
        "internal_payload": (current_values["fuselage_body_length_m"] - 1 - (0.5*hard_constraints["internal_payload_length"]), 0.35*current_values["fuselage_body_height_m"]),
        "wing_payload": (current_values["wing_le_position_m"] + 0.5 * current_values["chord_m"], -0.2*current_values["fuselage_body_height_m"]),
        "wing_fuel": (current_values["wing_le_position_m"] + 0.45 * current_values["chord_m"], assumed_and_set["wing_airfoil_thickness_ratio"] * current_values["chord_m"]),
        "fuselage_fuel": (current_values["wing_le_position_m"] - (0.5*assumed_and_set["fuselage_fuel_tank_length"]), 0.35*current_values["fuselage_body_height_m"]),
        "avionics": (current_values["fuselage_body_length_m"] * 0.1, 0.2*current_values["fuselage_body_height_m"]),
        "landing_gear": (0.7 * current_values["fuselage_body_length_m"], -0.1 * current_values["fuselage_body_height_m"]),
        "misc": (cg_estimate, 0.5 * current_values["fuselage_body_height_m"]),
    }

    cg_x_dict = {k: v[0] for k, v in full_dict.items()}
    cg_z_dict = {k: v[1] for k, v in full_dict.items()}

    return (cg_x_dict, cg_z_dict)

def calculate_eta_h(current_values, phase='cruise'):
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


def cross_section_area_fuselage_fuel_tank(fuselage_diameter, diameter_ratio_used_for_fuel_tank):
    """
    Calculate the area of a rectangle inscribed in a circle based on width ratio.

    Args:
        diameter (float): Diameter of the circle.
        width_ratio (float): Ratio of rectangle width to circle diameter (e.g., 0.8 for 80%).

    Returns:
        float: Area of the rectangle.
    """
    if not (0 <= diameter_ratio_used_for_fuel_tank <= 1):
        raise ValueError("width_ratio must be between 0 and 1.")

    # Compute half-width of rectangle
    half_width = (diameter_ratio_used_for_fuel_tank * fuselage_diameter) / 2

    # Radius of the circle
    R = fuselage_diameter / 2

    # Vertical distance from center to rectangle side
    y = math.sqrt(R**2 - half_width**2)

    # Rectangle height is double this vertical distance
    height = 2 * y

    # Rectangle width
    width = diameter_ratio_used_for_fuel_tank * fuselage_diameter

    # Area
    area = width * height

    return area


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
    high_wing_offset_m = assumed_and_set.get("high_wing_offset_m", 0.0)

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

def get_wing_or_tail_mass(span, cross_section_perimeter_length, tail_or_wing, assumed_and_set):

    
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

    total_wing_mass = total_skin_mass + total_core_mass

    return total_wing_mass



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

    alpha_min = -5
    alpha_max = 10

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

    coeff_ratio = get_cl_alpha_at(deflections_dict[f"{phase_for_delta}_0"], current_values['cruiseout_angle_of_attack_deg']  + assumed_and_set["wing_incident_angle"], "CM", 0.1) / get_cl_alpha_at(deflections_dict[f"{phase_for_delta}_0"], current_values['cruiseout_angle_of_attack_deg']  + assumed_and_set["wing_incident_angle"], "CL", 0.1)
    # new_wing_ac = wing_ac - (cl_row['CM']/cl_row['CL']) * current_values["chord_m"]
    new_wing_ac = wing_ac - coeff_ratio * current_values["chord_m"]

    eta_h = calculate_eta_h(current_values, phase=phase)

    # neutral_point_m = new_wing_ac + Vh * current_values['chord_m']
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