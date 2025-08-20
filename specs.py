
import math
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from ADRpy import atmospheres as at

## engine specs - equivalent to Rotax 912 UL/A/F from china's zosen
## link https://www.austars-model.com/zonsen-c80-ca300-aero-engine-carby-version-2000h-tbo-1200cc-59kw80hp-dcdi-power-we-starter_g24276.html

engine_specs = {
    "price": 13000,  # USD
    "dry_weight_kg": 70,  # kg
    "max_power_kw": 59.6, # kW
    "max_rpm": 5800, # rpm
    "max_continuous_power_kw": 58, # kW
    "max_continuous_rpm": 5500, # rpm
    "max_torque_nm": 103, # Nm
    "power_to_weight_ratio": 0.96,  # kW/kg
    "fuel_type": "gasoline 95 octane",
    "fuel_density": 0.74,  # kg/L
    "specific_fuel_consumption": 0.285,  # kg/kWh
    "reduction_ratio": 2.43,  # dimensionless
    "gear_box_efficiency": 0.95,  # dimensionless
}

engine_altitude_performance = { # power in kw vs altitude in km
    "0" : 58.5, # kW
    "1000" : 52, # kW
    "2000" : 46.5, # kW
    "3000" : 41, # kW
    "4000" : 35.5, # kW
    "5000" : 32, # kW
}

engine_power_to_rpm = { # power in kw vs rpm
    "3000" : 28.5, # kW
    "3500" : 37, # kW
    "4000" : 44, # kW
    "4500" : 51.5, # kW
    "5000" : 56, # kW
    "5500" : 58, # kW
}

engine_sfc_to_rpm = { # sfc in kg/kWh vs rpm
    "3000" : 0.42, # kg/kWh
    "3500" : 0.36, # kg/kWh
    "4000" : 0.325, # kg/kWh
    "4500" : 0.305, # kg/kWh
    "5000" : 0.27, # kg/kWh
    "5500" : 0.285, # kg/kWh
}

## propeller specs - equivalent to rotax prop 3BO
## link https://www.sensenich.com/shop/aircraft/3-blade-rotax-ground-adjustable-propeller/
## https://kievprop.com/

propeller_specs = {
    "price": 2000,  # USD
    "diameter_m": 1.87,  # m
    "blade_count": 3,  # dimensionless
    "weight_kg": 4,  # kg
    "moment_of_inertia_kgm2": 0.8,  # kg*m^2
    "max_rpm": 2600,  # rpm
    "max_thrust_kgf" : 250, # kgf
    "max_thrust_n": 2450,  # N
    "efficiency": {
        'take-off': 0.65,        # High RPM, static thrust conditions — realistic
        'climb': 0.75,           # Prop operating at high power, moderate efficiency
        'cruise': 0.8,           # Most efficient phase — optimized for this

        'turn': 0.78,            # Similar to cruise, slight variation
        'servceil': 0.7          # Efficiency drops at high altitude due to low air density
    }
}

airfoil_selig_1223_data = {
    'CDTO': 0.05,                # Slightly higher due to flaps, draggy gear, and possibly pusher config
    'CLTO': 0.6,                 # Moderate lift at rotation
    'CLmaxTO': 1.8,              # With flaps, achievable and gives safety margin for 50 m takeoff
    'CLmaxclean': 1.45,          # Matches a modest airfoil in clean config (no flaps)
    'CLminclean': -1.0,
    'CLslope': 6.28,             # Lift curve slope (1/rad)
    'CDminclean': 0.03
}

sandwich_specs = {
    "fuselage": {
        "total_thickness_m": 0.008,
        "core_thickness_m": 0.006
    },
    "wing": {
        "total_thickness_m": 0.012,
        "core_thickness_m": 0.010
    },
    "tail": {
        "total_thickness_m": 0.006,
        "core_thickness_m": 0.004
    },
    "control_surface": {
        "total_thickness_m": 0.005,
        "core_thickness_m": 0.003
    }
}
