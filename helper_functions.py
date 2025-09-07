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

def calc_required_wing_area(mtow_kg, cl_max, rho, v_stall_kmh):

    v_stall_ms = kmh_to_ms(v_stall_kmh)
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
        "internal_payload": ((0.5*hard_constraints["internal_payload_length"]) + 1.5, 
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

def _closed_form_trim_analysis(current_values, assumed_and_set, hard_constraints, deflections_dict, phase):
    """
    Computes wing CL, tail CL, wing/tail AoA and required elevator deflection
    for a given flight phase using closed-form trim equations.
    """
    phase_for_delta = ""
    # --- Normalise phase naming ---
    if ("cruise" in phase) or (phase == "loiter"):
        phase_for_delta = "cruise"
    elif phase in ["takeoff", "landing"]:
        phase_for_delta = "takeoff"

    # --- Pull values from dicts ---
    m = current_values["mtow"]  # or use a per-phase mass if stored separately
    rho = get_air_density(hard_constraints["cruise_altitude_m"])
    V_ms = kmh_to_ms(current_values[f"{phase}_speed_kmh"])
    Sw = current_values["wing_area_m2"]
    St = current_values["horizontal_tail_area_m2"]
    cbar = current_values["chord_m"]
    lt = current_values["tail_arm_m"]

    # static margin (fraction of MAC)
    SM = current_values[f"{phase}_static_margin"]

    qt_over_q = current_values.get("tail_dynamic_pressure_ratio", 1.0)
    downwash_grad = assumed_and_set.get("ht_downwash_efficiency_coeff", 0.3)
    wing_inc = assumed_and_set["wing_incident_angle"]
    ht_inc = assumed_and_set["ht_incident_angle"]

    # --- Equilibrium equations ---
    q = 0.5 * rho * V_ms**2
    qt = qt_over_q * q
    W = m * 9.81
    CL_req = W / (q * Sw)

    r = (qt / q) * (St / Sw)         # relative tail area*q
    k = SM * (cbar / lt)

    # Wing CL
    CL_w = CL_req / (1.0 - k)

    delta_cl_from_thrust = get_delt_cl_from_thrust(current_values, hard_constraints, phase)

    # Tail CL (no thrust moment term here; add if you model it)
    CL_t = - k * (1.0 / r) * CL_w + delta_cl_from_thrust

    # --- Wing AoA from polar ---
    alpha_w = get_row_for_cl(deflections_dict[f"{phase_for_delta}_0"], CL_w)["alpha"] - wing_inc

    # --- Tail AoA seen by the tail (include downwash) ---
    alpha_t = alpha_w * (1.0 - downwash_grad) + ht_inc

    # --- Interpolate to find elevator deflection needed for CL_t ---

    # delta_e = get_row_for_cl(df, CL_t)["alpha"]
    delta_e = solve_delta_e_for_CLt(CL_t, alpha_t, deflections_dict, phase=phase_for_delta)

    return {
        "cl_wing": CL_w,
        "cl_tail_required": CL_t,
        "alpha_wing_deg": alpha_w,
        "alpha_tail_deg": alpha_t,
        "delta_elevator_deg": delta_e,
    }

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
    eta_h = current_values.get("tail_dynamic_pressure_ratio", 1.0)           # q_t / q

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

    CL_t = - (Cm_w + CL_w * d_w + Cm_fuse + Cm_thrust) / denom

    # Optional: refine CL_w by removing the tail lift from force balance (one correction step)
    L_tail = (eta_h * q) * Sh * CL_t
    CL_w = (W - L_tail) / (q * Sw)
#06-5188286
    # Re-read wing polar and Cm if you want a second, tighter pass:
    row_w   = get_row_for_cl(deflections_dict[f"{phase_for_delta}_0"], CL_w)
    alpha_w = float(row_w["alpha"]) - i_w
    Cm_w    = float(row_w["CM"])
    alpha_t = (1.0 - deda) * alpha_w + i_h
    # (You could recompute CL_t again with the updated CL_w if desired.)

    # --- Elevator deflection to achieve CL_t at alpha_t ---
    delta_e = solve_delta_e_for_CLt(CL_t, alpha_t, deflections_dict, phase=phase_for_delta)

    return {
        "cl_wing": CL_w,
        "cl_tail_required": CL_t,
        "alpha_wing_deg": alpha_w,   # body AoA
        "alpha_tail_deg": alpha_t,   # tail local geometric AoA
        "delta_elevator_deg": delta_e,
    }


def required_wing_area_consistent(cur, assumed_and_set, hard_constraints,
                                  weights_dict_kg_no_fuel,
                                  phase):
    """
    Return the required wing area [m^2] for the given phase, consistently:
      - takeoff:  S = W_TO / (q_TO * CL_max)           (tail ignored at stall)
      - cruise/loiter/back: S solves W = q*(Sw*CLw + q*eta_h*Sh*CLt),
        using Sh/Sw from tail volume: Sh/Sw = Vh * c_bar / l_t

    Expects in `cur` (from your iteration): 
      speeds per phase, chord_m, tail_arm_m, 
      cl per phase (wing), cl_tail_required per phase (tail), etc.
    """

    # --- helpers ---
    def phase_weight_kg(phase):
        # payload is zero on cruiseback, present otherwise
        payload_internal = weights_dict_kg_no_fuel.get("internal_payload", 0.0) if phase != "cruiseback" else 0.0
        payload_wing     = weights_dict_kg_no_fuel.get("wing_payload", 0.0)     if phase != "cruiseback" else 0.0

        return (weights_dict_kg_no_fuel["fuselage"] +
                weights_dict_kg_no_fuel["wing"] +
                weights_dict_kg_no_fuel["tails"] +
                weights_dict_kg_no_fuel["engine"] +
                weights_dict_kg_no_fuel["propeller"] +
                weights_dict_kg_no_fuel["avionics"] +
                weights_dict_kg_no_fuel["landing_gear"] +
                weights_dict_kg_no_fuel["misc"] +
                cur.get("fuselage_fuel", 0.0) +
                cur.get("wing_fuel", 0.0) +
                payload_internal + payload_wing)

    def q_dyn(alt_m, V_kmh):
        rho = get_air_density(alt_m)
        V   = kmh_to_ms(V_kmh)
        return 0.5 * rho * V**2

    # --- phase setup ---
    if phase.lower() in ("takeoff", "stall"):
        q = q_dyn(0.0, hard_constraints["stall_speed_kmh"])
        W = phase_weight_kg("takeoff") * g
        CLmax = hard_constraints["CL_max"]
        if CLmax <= 0:
            raise ValueError("CL_max must be > 0 for takeoff sizing.")
        return W / (q * CLmax)

    # cruise/loiter/cruiseback: include tail contribution via Vh
    phase = phase.lower()
    if phase not in ("cruiseout", "loiter", "cruiseback"):
        raise ValueError(f"Unknown phase '{phase}'.")

    # dynamic pressure at cruise altitude with phase speed
    q = q_dyn(hard_constraints["cruise_altitude_m"], cur[f"{phase}_speed_kmh"])
    W = phase_weight_kg(phase) * g

    # wing & tail CLs from your converged state
    CLw = cur.get(f"{phase}_cl", None)
    CLt = cur.get(f"{phase}_cl_tail_required", 0.0)  # default 0 if missing
    if CLw is None:
        raise KeyError(f"{phase}_cl not found in `cur`.")

    # geometry/volume terms
    Vh   = assumed_and_set["horizontal_tail_volume_coefficient"]  # (l_t * S_h) / (S_w * c_bar)
    lt   = cur["tail_arm_m"]
    cbar = cur["chord_m"]
    if lt <= 0 or cbar <= 0:
        raise ValueError("tail_arm_m and chord_m must be > 0.")

    # Sh/Sw from Vh and current geometry (no need to know Sw explicitly)
    Sh_over_Sw = Vh * cbar / lt

    # dynamic pressure ratio at tail
    eta_h = calculate_eta_h(cur, phase=phase)

    # denominator for Sw; guard against non-physical values
    denom = (CLw + eta_h * Sh_over_Sw * CLt)
    if denom <= 1e-9:
        # If tail downforce is very large (or CLw tiny), you could hit denom≈0.
        # In that edge case, ignore tail for sizing to avoid division blow-up.
        denom = max(CLw, 1e-6)

    return W / (q * denom)


def effective_CLmax_partial_span(
    deflections_dict,
    phase_prefix="takeoff",            # "takeoff" polars are preferred; fall back to "cruise"
    flap_deflection_deg=20,
    flap_span_fraction=0.50,          # b_flap / b_wing
    spanwise_flap_effectiveness=0.95, # ≤1.0
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

import numpy as np

def _sorted_map_arrays(engine_power_to_rpm, engine_sfc_to_rpm):
    """Convert your dicts (string keys) into sorted numeric arrays by RPM."""
    # Ensure numeric, sorted by rpm
    rpms = np.array(sorted([int(k) for k in engine_power_to_rpm.keys()]), dtype=float)
    PkW = np.array([float(engine_power_to_rpm[str(int(r))]) for r in rpms], dtype=float)
    SFC = np.array([float(engine_sfc_to_rpm[str(int(r))])     for r in rpms], dtype=float)
    # Sanity (same RPM grid)
    if len(SFC) != len(PkW):
        raise ValueError("Power and SFC maps have different lengths.")
    return rpms, PkW, SFC

def make_engine_interpolants(engine_power_to_rpm, engine_sfc_to_rpm):
    """
    Returns callables:
      P_of_rpm(rpm) [kW], SFC_of_rpm(rpm) [kg/kWh], rpm_for_power(P_req_kw) [rpm]
    Monotonic piecewise-linear; extrapolation is clamped to map edges.
    """
    rpms, PkW, SFC = _sorted_map_arrays(engine_power_to_rpm, engine_sfc_to_rpm)

    # Forward: P(rpm), SFC(rpm)
    def _interp_clamped(x, x_arr, y_arr):
        x = np.asarray(x, dtype=float)
        x_clamped = np.clip(x, x_arr[0], x_arr[-1])
        return np.interp(x_clamped, x_arr, y_arr)

    def P_of_rpm(rpm):
        return _interp_clamped(rpm, rpms, PkW)

    def SFC_of_rpm(rpm):
        return _interp_clamped(rpm, rpms, SFC)

    # Inverse: rpm(P) — guard monotonicity
    if not np.all(np.diff(PkW) > 0.0):
        # If the power curve has flats/noise, add tiny epsilon to enforce strict monotone
        eps = 1e-6 * np.arange(len(PkW))
        PkW_mono = PkW + eps
    else:
        PkW_mono = PkW.copy()

    def rpm_for_power(P_req_kw):
        """Minimum RPM that can supply at least P_req_kw; clamps to ends."""
        P = float(P_req_kw)
        if P <= PkW_mono[0]:
            return rpms[0]
        if P >= PkW_mono[-1]:
            return rpms[-1]
        return float(np.interp(P, PkW_mono, rpms))

    return P_of_rpm, SFC_of_rpm, rpm_for_power

def engine_available_power_kw_at_alt(rpm, engine_specs, sigma, use_max=True):
    """
    Apply altitude derate to map power (sea-level map → altitude).
    If turbo_normalized up to critical altitude, hold power.
    engine_specs expects:
      - "altitude_power_exponent" (default 1.0)
      - "turbo_normalized" (bool), "critical_altitude_m"
    """
    expo = float(engine_specs.get("altitude_power_exponent", 1.0))
    turbo = bool(engine_specs.get("turbo_normalized", False))
    crit  = float(engine_specs.get("critical_altitude_m", 0.0))
    # rpm power at sea level:
    # caller provides base P_of_rpm(rpm) separately; this function only scales with σ if needed
    # We keep this scaler here for clarity:
    if turbo and sigma < 1.0 and crit > 0:
        # Caller should pass sigma at the altitude. If below crit-alt, sigma≈1 scaling (i.e., no derate).
        pass
    scale = 1.0 if (turbo and crit > 0 and sigma < 1.0) else (sigma ** expo)
    return scale  # multiply this on top of sea-level P_of_rpm(rpm)

def choose_engine_setting_for_climb(cur, V_ms, altitude_m, mode, 
                                    engine_specs, propeller_specs,
                                    P_of_rpm, SFC_of_rpm, rpm_for_power,
                                    eta_gb=1.0, sigma=None, P_req_air_w=None,
                                    rpm_limits=(3000.0, 5500.0)):
    """
    Decide RPM and compute:
      - P_shaft_kw at that RPM (after altitude derate),
      - SFC (kg/kWh) at that RPM,
      - P_avail_to_air_w = η_p(V)*η_gb*P_shaft_kw*1000
    Modes:
      - "max_rate": pick rpm_max → max P_shaft → best ROC.
      - "min_sfc_at_required": deliver at least P_req_air_w to the AIR and minimize SFC.
        Requires P_req_air_w (from drag*speed) and prop efficiency for conversion.
    """
    # Atmosphere scaling
    if sigma is None:
        rho = get_air_density(altitude_m)
        sigma = rho / get_air_density(0.0)

    # Prop efficiency vs V: reuse your linear blend approach
    eta_takeoff = propeller_specs["efficiency"].get("takeoff", propeller_specs["efficiency"].get("take-off", 0.75))
    eta_cruise  = propeller_specs["efficiency"].get("cruise", eta_takeoff)
    Vc_ms = kmh_to_ms(cur.get("cruiseout_speed_kmh", 150.0))
    t = max(0.0, min(1.0, V_ms / max(Vc_ms, 1.0)))
    eta_p = (1.0 - t) * eta_takeoff + t * eta_cruise

    rpm_min, rpm_max = rpm_limits

    if mode == "max_rate":
        rpm = rpm_max
        P_sea_kw = P_of_rpm(rpm)
        scale = engine_available_power_kw_at_alt(rpm, engine_specs, sigma)
        P_shaft_kw = P_sea_kw * scale
        sfc = SFC_of_rpm(rpm)
        P_air_w = eta_p * eta_gb * P_shaft_kw * 1000.0
        return {"rpm": rpm, "P_shaft_kw": P_shaft_kw, "sfc_kg_per_kwh": sfc,
                "eta_p": eta_p, "P_avail_air_w": P_air_w}

    elif mode == "min_sfc_at_required":
        if P_req_air_w is None:
            raise ValueError("P_req_air_w must be provided for mode='min_sfc_at_required'")
        # Required shaft power to deliver P_req_air_w to AIR:
        P_req_shaft_kw = P_req_air_w / max(eta_p * eta_gb, 1e-6) / 1000.0
        # Find the minimum RPM that can deliver ≥ this shaft power (after altitude derate).
        # We’ll iterate a bit because derate depends on sigma^expo but not on rpm in our model.
        rpm_guess = rpm_for_power(P_req_shaft_kw)  # sea level
        rpm = float(np.clip(rpm_guess, rpm_min, rpm_max))
        # Ensure enough after derate; if not, push to higher rpm
        for _ in range(3):
            P_sea_kw = P_of_rpm(rpm)
            scale = engine_available_power_kw_at_alt(rpm, engine_specs, sigma)
            if P_sea_kw * scale >= P_req_shaft_kw or rpm >= rpm_max - 1.0:
                break
            # push rpm up proportionally
            rpm = float(np.clip(rpm + 200.0, rpm_min, rpm_max))

        # Among feasible RPMs (± a small neighborhood), pick the one with min SFC
        candidates = np.clip(np.array([rpm-200, rpm, rpm+200, rpm+400]), rpm_min, rpm_max)
        best = None
        for r in np.unique(candidates):
            P_sea_kw = P_of_rpm(r)
            scale = engine_available_power_kw_at_alt(r, engine_specs, sigma)
            P_shaft_kw = P_sea_kw * scale
            if P_shaft_kw < P_req_shaft_kw: 
                continue
            sfc = SFC_of_rpm(r)
            if (best is None) or (sfc < best["sfc_kg_per_kwh"]):
                best = {"rpm": float(r), "P_shaft_kw": float(P_shaft_kw), "sfc_kg_per_kwh": float(sfc)}
        if best is None:
            # Not enough power at any rpm → fall back to max rpm (infeasible warning handled upstream)
            r = rpm_max
            P_shaft_kw = P_of_rpm(r) * engine_available_power_kw_at_alt(r, engine_specs, sigma)
            best = {"rpm": r, "P_shaft_kw": P_shaft_kw, "sfc_kg_per_kwh": SFC_of_rpm(r)}

        best["eta_p"] = eta_p
        best["P_avail_air_w"] = eta_p * eta_gb * best["P_shaft_kw"] * 1000.0
        return best

    else:
        raise ValueError("mode must be 'max_rate' or 'min_sfc_at_required'")


def compute_time_to_altitude_with_engine_maps(
    cur, assumed_and_set, hard_constraints, deflections_dict,
    engine_power_to_rpm, engine_sfc_to_rpm,
    target_altitude_m=None, start_altitude_m=0.0, altitude_slices=24,
    climb_phase_prefix="takeoff",          # "takeoff" (flaps) or "cruise" (clean)
    operate_mode="max_rate",               # "max_rate" or "min_sfc_at_required"
    include_tail_drag=True
):
    """
    March to the target altitude; at each slice, find Vy by maximizing ROC(V).
    Uses your power/SFC vs RPM maps.

    Returns:
      { "time_to_altitude_s": ..., "profile": [per-slice dicts], "roc_curves": {...} }
    """
    import math
    g = 9.81

    # Build engine interpolants (power & SFC vs RPM, plus inverse)
    P_of_rpm, SFC_of_rpm, rpm_for_power = make_engine_interpolants(engine_power_to_rpm, engine_sfc_to_rpm)

    # Geometry/airframe data
    wing_planform_area_m2 = float(cur["wing_area_m2"])
    horizontal_tail_area_m2 = float(cur["horizontal_tail_area_m2"])
    mean_aerodynamic_chord_m = float(cur["chord_m"])
    tail_arm_m = float(cur["tail_arm_m"])
    aspect_ratio_wing = float(assumed_and_set.get("aspect_ratio", 12.0))
    aspect_ratio_tail = float(assumed_and_set.get("AR_horizontal", 4.0))
    oswald_wing = 0.85
    oswald_tail = 0.80
    induced_k_wing = 1.0 / (math.pi * aspect_ratio_wing * oswald_wing)
    induced_k_tail = 1.0 / (math.pi * aspect_ratio_tail * oswald_tail)

    # Incidence and downwash
    wing_incidence_deg = float(assumed_and_set.get("wing_incident_angle", 0.0))
    tail_incidence_deg = float(assumed_and_set.get("ht_incident_angle",   0.0))
    downwash_deda = float(assumed_and_set.get("ht_downwash_efficiency_coeff", 0.30))
    one_minus_deda = 1.0 - downwash_deda

    # Mass at climb start (you can add fuel burn inside the loop if desired)
    mass_kg = cur.get("climb_mass_kg", cur.get("mtow"))
    weight_N = mass_kg * g

    # Phase-dependent CG and wing AC (fallbacks)
    xcg_m = cur.get("climb_cg_from_nose_m", cur.get("cruiseout_cg_from_nose_m"))
    x_ac_wing_m = cur.get("climb_x_ac_w_m", cur.get("cruiseout_x_ac_w_m",
                        cur["wing_le_position_m"] + 0.25 * mean_aerodynamic_chord_m))

    # Engine/prop efficiencies & altitude derate
    eta_gbx = float(engine_specs.get("gear_box_efficiency", 1.0))
    eta_takeoff = propeller_specs["efficiency"].get("takeoff", propeller_specs["efficiency"].get("take-off", 0.75))
    eta_cruise  = propeller_specs["efficiency"].get("cruise", eta_takeoff)

    turbo_norm = bool(engine_specs.get("turbo_normalized", False))
    critical_alt_m = float(engine_specs.get("critical_altitude_m", 0.0))
    power_exponent_sigma = float(engine_specs.get("altitude_power_exponent", 1.0))

    def density_ratio_sigma(h_m):
        return get_air_density(h_m) / get_air_density(0.0)

    def altitude_power_scale(sigma):
        if turbo_norm and critical_alt_m > 0:
            # Flat to critical altitude (simple model)
            return 1.0
        return sigma ** power_exponent_sigma

    def prop_efficiency_vs_speed(V_ms):
        V_cruise_ms = kmh_to_ms(cur.get("cruiseout_speed_kmh", 150.0))
        t = max(0.0, min(1.0, V_ms / max(1.0, V_cruise_ms)))
        return (1.0 - t) * eta_takeoff + t * eta_cruise

    # Clean CD0 baseline for the chosen climb phase
    try:
        CD0_phase = float(get_row_for_cl(deflections_dict[f"{climb_phase_prefix}_0"], 0.0)["CD"])
    except Exception:
        CD0_phase = float(get_row_for_cl(deflections_dict["cruise_0"], 0.0)["CD"])

    # Wing-tail trim & drag at (V, h)
    def trimmed_drag_and_power_required(V_ms, altitude_m):
        rho = get_air_density(altitude_m)
        Q = 0.5 * rho * V_ms * V_ms

        # Initial guess: wing CL from lift = weight; refine with tail moment balance
        CL_wing = weight_N / (Q * wing_planform_area_m2)

        # Under-relaxed two-step: (wing CM + thrust CM) balanced by tail lift moment
        for _ in range(3):
            # Find wing α that yields CL_wing on the selected phase polar (clean if "cruise", flapped if "takeoff")
            alpha_wing_deg = get_row_for_cl(deflections_dict[f"{climb_phase_prefix}_0"], CL_wing)["alpha"] - wing_incidence_deg

            # Tail local α including downwash
            alpha_tail_deg = one_minus_deda * alpha_wing_deg + tail_incidence_deg

            # Wing section CM from polar at that α (phase polar)
            row_w = get_coefficients_at_alpha(deflections_dict[f"{climb_phase_prefix}_0"], alpha_wing_deg + wing_incidence_deg)
            Cm_wing = float(row_w["CM"])

            # Thrust moment about CG (vertical offset): sign nose-down if thrust above CG
            thrustline_z_m = float(cur.get("thrustline_z_from_floor_m", 0.0))
            z_cg_m = float(cur.get("climb_cg_from_floor_m", cur.get("cruiseout_cg_from_floor_m", 0.0)))
            # Use power available estimate to the AIR for the moment; small effect on CLt
            eta_p = prop_efficiency_vs_speed(V_ms)
            # Pick an RPM near the top (max-rate default)
            rpm_top = max(int(max(engine_power_to_rpm.keys(), key=int)), 1)
            P_sea_kw = float(engine_power_to_rpm[str(rpm_top)])
            Pav_kw = P_sea_kw * altitude_power_scale(density_ratio_sigma(altitude_m))
            Pav_w  = Pav_kw * 1000.0 * eta_p * eta_gbx
            T_est  = Pav_w / max(V_ms, 1.0)
            Cm_thrust = - T_est * (thrustline_z_m - z_cg_m) / (Q * wing_planform_area_m2 * mean_aerodynamic_chord_m)

            nondim_arm_w = (x_ac_wing_m - xcg_m) / mean_aerodynamic_chord_m
            tail_q_ratio = calculate_eta_h(cur, phase="cruiseout") if "calculate_eta_h" in globals() else 1.0
            denom = tail_q_ratio * (horizontal_tail_area_m2/wing_planform_area_m2) * (tail_arm_m/mean_aerodynamic_chord_m)
            CL_tail = - (Cm_wing + CL_wing*nondim_arm_w + Cm_thrust) / max(denom, 1e-9)

            # Correct wing CL by subtracting tail lift
            tail_lift_N = tail_q_ratio * Q * horizontal_tail_area_m2 * CL_tail
            CL_wing_new = (weight_N - tail_lift_N) / (Q * wing_planform_area_m2)
            CL_wing = 0.65*CL_wing + 0.35*CL_wing_new

        # Drag (wing + tail)
        CD_wing = CD0_phase + induced_k_wing * (CL_wing**2)
        if include_tail_drag:
            CD_tail = CD0_phase + induced_k_tail * (CL_tail**2)
        else:
            CD_tail = 0.0

        total_drag_N = Q * (wing_planform_area_m2 * CD_wing + horizontal_tail_area_m2 * CD_tail)
        power_required_W = total_drag_N * V_ms

        return power_required_W

    # Altitude march (best-rate at each slice)
    H_target = float(target_altitude_m if target_altitude_m is not None else hard_constraints.get("cruise_altitude_m", 3000.0))
    if H_target <= start_altitude_m:
        return {"time_to_altitude_s": 0.0, "profile": []}

    ΔH = (H_target - start_altitude_m) / altitude_slices
    total_time_s = 0.0
    profile = []
    roc_curves = {}

    for i in range(altitude_slices):
        H_mid = start_altitude_m + (i + 0.5) * ΔH
        rho = get_air_density(H_mid)
        sigma = rho / get_air_density(0.0)

        # Speed sweep (bounds from stall guess to cruise)
        # Use CL_max of phase to estimate Vs
        CLmax_phase = float(deflections_dict[f"{climb_phase_prefix}_0"]["CL"].max())
        Vs = (2.0 * weight_N / (rho * wing_planform_area_m2 * CLmax_phase)) ** 0.5
        V_lo = max(1.05*Vs, 0.6*Vs)
        V_hi = max(kmh_to_ms(cur.get("cruiseout_speed_kmh", 150.0)), 1.6*Vs)

        best_ROC = -1.0
        best_V = None
        curve_V, curve_ROC = [], []

        for frac in np.linspace(0.0, 1.0, 60):
            V = V_lo + frac * (V_hi - V_lo)
            power_req_W = trimmed_drag_and_power_required(V, H_mid)

            # Choose RPM & compute available power to the AIR + SFC (from your maps)
            # For "max_rate", we run near the top end of the rpm map
            if operate_mode == "max_rate":
                rpm_cmd = float(max(int(max(engine_power_to_rpm.keys(), key=int)), 1))
                P_sea_kw = P_of_rpm(rpm_cmd)
                P_shaft_kw = P_sea_kw * altitude_power_scale(sigma)
                eta_p = prop_efficiency_vs_speed(V)
                power_avail_W = eta_p * eta_gbx * P_shaft_kw * 1000.0

            else:  # "min_sfc_at_required": deliver at least required power with lowest SFC
                eta_p = prop_efficiency_vs_speed(V)
                P_req_shaft_kw = power_req_W / max(eta_p * eta_gbx, 1e-6) / 1000.0
                rpm_guess = rpm_for_power(P_req_shaft_kw)
                rpm_cmd = float(np.clip(rpm_guess, min(P_of_rpm.__defaults__[0]), max(P_of_rpm.__defaults__[0])))
                # simple correction for altitude scale:
                # search a small neighborhood for feasibility and min SFC
                candidates = np.clip(np.array([rpm_cmd-200, rpm_cmd, rpm_cmd+200, rpm_cmd+400]),
                                     min(engine_power_to_rpm, key=int),
                                     max(engine_power_to_rpm, key=int)).astype(float)
                best = None
                for r in np.unique(candidates):
                    P_sea_kw = P_of_rpm(r)
                    P_shaft_kw = P_sea_kw * altitude_power_scale(sigma)
                    if P_shaft_kw < P_req_shaft_kw: 
                        continue
                    sfc = SFC_of_rpm(r)
                    if best is None or sfc < best["sfc"]:
                        best = {"rpm": r, "P_shaft_kw": P_shaft_kw, "sfc": sfc}
                if best is None:
                    # not enough power—fall back to max rpm (ROC likely small)
                    r = float(max(int(max(engine_power_to_rpm.keys(), key=int)), 1))
                    P_shaft_kw = P_of_rpm(r) * altitude_power_scale(sigma)
                    rpm_cmd = r
                else:
                    P_shaft_kw = best["P_shaft_kw"]; rpm_cmd = best["rpm"]
                power_avail_W = eta_p * eta_gbx * P_shaft_kw * 1000.0

            ROC = max(0.0, (power_avail_W - power_req_W) / weight_N)
            curve_V.append(V); curve_ROC.append(ROC)
            if ROC > best_ROC:
                best_ROC = ROC; best_V = V

        if best_ROC <= 1e-6:
            return {"time_to_altitude_s": float("inf"),
                    "failed_at_m": H_mid, "profile": profile, "roc_curves": roc_curves}

        dt = ΔH / best_ROC
        total_time_s += dt
        profile.append({"alt_m": H_mid, "Vy_ms": best_V, "ROC_ms": best_ROC})
        roc_curves[H_mid] = {"V_ms": curve_V, "ROC_ms": curve_ROC}

    return {"time_to_altitude_s": total_time_s, "profile": profile, "roc_curves": roc_curves}

def size_wing_area_for_takeoff_with_flaps(
    cur, assumed_and_set, hard_constraints, deflections_dict,
    flap_deflection_deg=20, flap_span_fraction=0.50,
    spanwise_flap_effectiveness=0.95,
    rotation_factor_Vlof_over_Vstall=1.20,   # V_LOF = 1.2 * V_stall_TO
    ground_CL_fraction_of_CLmax=0.90,        # average CL during roll
    oswald_e_takeoff=0.80,
    max_bisection_iter=30, area_tol=0.5
):
    """
    Returns a dict with:
      - wing_area_required_m2 (max of stall and distance constraints)
      - wing_area_from_stall_m2
      - wing_area_from_groundrun_m2
      - takeoff_ground_run_m
      - V_stall_TO_ms, V_LOF_ms
      - details (CD0_TO, CLmax_TO_eff, etc.)
    """
    import math
    g = 9.81
    sea_level_density = get_air_density(0.0)
    rolling_mu = float(assumed_and_set.get("rolling_resistance_coefficient", 0.04))
    required_ground_run_m = float(hard_constraints.get("takeoff_distance_max_m", 50.0))

    # Mass/weight at takeoff
    takeoff_mass_kg = cur.get("takeoff_mass_kg", cur.get("mtow"))
    weight_N = takeoff_mass_kg * g

    # Aspect ratio and induced drag factor
    AR_wing = float(assumed_and_set.get("aspect_ratio", 12.0))
    k_induced = 1.0 / (math.pi * AR_wing * oswald_e_takeoff)

    # Effective CLmax for partial-span flaps (use *max* CL at the airfoil level, then blend)

    λ_eff = max(0.0, min(1.0, spanwise_flap_effectiveness * flap_span_fraction))
    clmax_pack = effective_CLmax_partial_span(
        deflections_dict,
        phase_prefix="takeoff",
        flap_deflection_deg=flap_deflection_deg,
        flap_span_fraction=flap_span_fraction,
        spanwise_flap_effectiveness=spanwise_flap_effectiveness,
        alpha_margin_deg=0.0  # or 0.5–1.0° if you want a conservative buffer
    )
    CLmax_TO_eff = clmax_pack["CLmax_eff"]
    alpha_at_CLmax_deg = clmax_pack["alpha_at_CLmax_deg"]

    # CD0 in takeoff config at CL≈0, blended
    def CD0_effective():
        def CD0_from(table_key):
            return float(get_row_for_cl(deflections_dict[table_key], 0.0)["CD"])
        CD0_clean = CD0_from("takeoff_0") if "takeoff_0" in deflections_dict else CD0_from("cruise_0")
        CD0_flap  = CD0_from(flap_key) if flap_key in deflections_dict else CD0_clean
        return (1.0 - λ_eff)*CD0_clean + λ_eff*CD0_flap
    CD0_TO = CD0_effective()

    # Stall-constraint area using YOUR stall speed requirement
    Vstall_target_ms = kmh_to_ms(float(hard_constraints.get("stall_speed_kmh", 55.0)))
    wing_area_from_stall_m2 = 2.0 * weight_N / (sea_level_density * Vstall_target_ms**2 * CLmax_TO_eff)

    # Ground-roll integrator with bisection on wing area
    def compute_ground_run_for_area(wing_planform_area_m2):
        V_stall_ms = math.sqrt(2.0 * weight_N / (sea_level_density * wing_planform_area_m2 * CLmax_TO_eff))
        V_liftoff_ms = rotation_factor_Vlof_over_Vstall * V_stall_ms
        dv = max(0.2, V_liftoff_ms / 400.0)
        distance_m = 0.0
        v = 0.1

        # Choose an average CL during roll; get α that matches the target CL on the effective polar
        CL_ground_avg = ground_CL_fraction_of_CLmax * CLmax_TO_eff
        alpha_ground_deg = solve_wing_alpha_for_target_CL_effective(
            deflections_dict, "takeoff", CL_ground_avg,
            flap_deflection_deg, flap_span_fraction, spanwise_flap_effectiveness
        )

        # Use **blended** CD at that alpha for parasitic part; add induced drag from CL_ground_avg
        coeffs = effective_wing_coefficients_at_alpha(
            deflections_dict, "takeoff", alpha_ground_deg,
            flap_deflection_deg, flap_span_fraction, spanwise_flap_effectiveness
        )
        CD_parasitic = coeffs["CD"]  # this already includes flap-induced profile increase at that α

        # Basic thrust model: P_air = η_p(V)*η_gb*P_shaft(RPM(V)); approximate η_p by your blend
        eta_takeoff = propeller_specs["efficiency"].get("takeoff", propeller_specs["efficiency"].get("take-off", 0.75))
        eta_gbx = float(engine_specs.get("gear_box_efficiency", 1.0))
        Pmax_kw = float(engine_specs.get("max_power_kw", engine_specs.get("max_cruise_power_kw", 0.0)))
        Pavail_air_W = eta_takeoff * eta_gbx * Pmax_kw * 1000.0  # nearly flat vs v at constant-power, low speed
        T_static_cap = float(engine_specs.get("static_thrust_N", 1e9))

        while v < V_liftoff_ms:
            Q = 0.5 * sea_level_density * v*v
            CD_total = CD_parasitic + k_induced * (CL_ground_avg**2)
            drag_N = Q * wing_planform_area_m2 * CD_total
            lift_N = Q * wing_planform_area_m2 * CL_ground_avg
            normal_N = max(0.0, weight_N - lift_N)

            # Thrust available ~ min(static cap, Pavail/V)
            T_avail = min(T_static_cap, Pavail_air_W / max(1.0, v))
            accel = (T_avail - drag_N - rolling_mu * normal_N) / takeoff_mass_kg
            if accel <= 1e-4:
                return float("inf"), V_stall_ms, V_liftoff_ms

            distance_m += (v / accel) * dv
            v += dv

        return distance_m, V_stall_ms, V_liftoff_ms

    # Bisection on area to meet ground-run limit
    area_lo = max(0.5 * wing_area_from_stall_m2, 0.1)
    area_hi = 4.0 * wing_area_from_stall_m2
    dist_hi, _, _ = compute_ground_run_for_area(area_hi)
    guard = 0
    while dist_hi == float("inf") and guard < 6:
        area_hi *= 1.5
        dist_hi, _, _ = compute_ground_run_for_area(area_hi)
        guard += 1

    wing_area_from_groundrun_m2 = area_hi
    for _ in range(max_bisection_iter):
        area_mid = 0.5 * (area_lo + area_hi)
        s_mid, _, _ = compute_ground_run_for_area(area_mid)
        if s_mid <= required_ground_run_m:
            wing_area_from_groundrun_m2 = area_mid
            area_hi = area_mid
        else:
            area_lo = area_mid
        if abs(area_hi - area_lo) < area_tol:
            break

    wing_area_required_m2 = max(wing_area_from_stall_m2, wing_area_from_groundrun_m2)
    s_final, Vstall_TO_ms, Vlof_ms = compute_ground_run_for_area(wing_area_required_m2)

    return {
        "wing_area_required_m2": wing_area_required_m2,
        "wing_area_from_stall_m2": wing_area_from_stall_m2,
        "wing_area_from_groundrun_m2": wing_area_from_groundrun_m2,
        "takeoff_ground_run_m": s_final,
        "V_stall_TO_ms": Vstall_TO_ms,
        "V_LOF_ms": Vlof_ms,
        "details": {
            "CLmax_TO_eff": CLmax_TO_eff,
            "CD0_TO_est": CD0_TO,
            "flap_span_fraction": flap_span_fraction,
            "flap_deflection_deg": flap_deflection_deg,
            "spanwise_flap_effectiveness": spanwise_flap_effectiveness
        }
    }
