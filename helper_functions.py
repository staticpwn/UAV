from scipy.interpolate import interp1d
import math
import numpy as np
from ADRpy import atmospheres as at
import ussa1976

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
        "tails": (current_values["tail_arm_m"] + current_values["wing_le_position_m"] + 0.25 * current_values["chord_m"], tail_local_cg_z + current_values["tail_boom_pylon_height_m"]),
        # "tails": 0.9*fuselage_length,
        "engine": (current_values["fuselage_body_length_m"] * 0.95, 0.5*current_values["fuselage_body_height_m"]),
        "propeller": (current_values["fuselage_body_length_m"] * 1.0, 0.5*current_values["fuselage_body_height_m"] ),
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
    wing_le_position = current_values["wing_le_position_m"]
    wing_chord = current_values["chord_m"]
    wing_ac = wing_le_position + 0.25 * wing_chord

    # 2. Horizontal Tail Aerodynamic Center (AC)
    # Tail LE is located at: cg + tail_arm - ht_chord*0.25 (assuming tail measured from cg)
    # But in your code: tail_arm is from CG to tail LE → so:
    cg_position = current_values["cruiseout_cg_from_nose_m"]
    tail_arm = current_values["tail_arm_m"]
    h_tail_chord = current_values["h_tail_chord_m"]
    
    h_tail_le_position = cg_position + tail_arm  # Tail LE relative to nose
    ht_ac = h_tail_le_position + 0.25 * h_tail_chord  # HT AC

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
    

