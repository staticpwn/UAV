import subprocess
import pandas as pd
import os



def save_airfoil_as_dat(filename, name, x, y):
    """
    Save airfoil coordinates to a .dat file in Selig format.

    Parameters:
        filename (str): Path to save the file (e.g., 'NACA2412.dat')
        name (str): Airfoil name to be written as the first line
        x (array-like): x-coordinates
        y (array-like): y-coordinates
    """
    with open(filename, 'w') as f:
        f.write(f"   {name}\n")
        for xi, yi in zip(x, y):
            f.write(f"  {xi:8.5f}     {yi:8.5f}\n")


def get_aerodynamic_coeffs_for_airfoil(correct_airfoil_name_with_space, Reynolds_number, from_dat=False):
    xfoil_path = os.path.join(os.getcwd(), r"..\XFOIL6.99\xfoil.exe")

    # Define the commands to be sent to XFOIL
    commands = f"""
    {correct_airfoil_name_with_space}
    OPER
    VISC {Reynolds_number}
    PACC
    {correct_airfoil_name_with_space}-{int(Reynolds_number)}.dat

    ASEQ -5 15 0.1

    QUIT
    """
    if from_dat:
        commands = f"""
    LOAD {correct_airfoil_name_with_space}-{int(Reynolds_number)}.dat
    PANE
    OPER
    VISC {Reynolds_number}
    PACC
    {correct_airfoil_name_with_space}-{int(Reynolds_number)}-polar.dat

    ASEQ -5 15 0.1

    QUIT
    """

    print(commands)
    # Write the commands to a temporary file
    with open('xfoil_input.txt', 'w') as f:
        f.write(commands)

    # Execute XFOIL with the input file
    with open('xfoil_input.txt', 'r') as input_file, open('xfoil_output.txt', 'w') as output_file:
        subprocess.run(xfoil_path, stdin=input_file, stdout=output_file)

def load_xfoil_polar(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_lines = []
    start_collecting = False

    for line in lines:
        line = line.strip()

        # Identify the line with column headers
        if line.startswith("alpha") and "CL" in line:
            start_collecting = True
            continue

        # Skip lines until data starts
        if start_collecting:
            # Stop if blank line or end of numeric data
            if not line or not any(char.isdigit() for char in line):
                continue

            # Try to parse the data
            try:
                parts = line.split()
                if len(parts) == 7:
                    alpha, cl, cd, cdp, cm, top_xtr, bot_xtr = map(float, parts)
                    data_lines.append([alpha, cl, cd, cdp, cm, top_xtr, bot_xtr])
            except ValueError:
                continue  # Skip malformed lines

    df = pd.DataFrame(data_lines, columns=["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"])
    return df

creationflags = subprocess.CREATE_NO_WINDOW
startupinfo = subprocess.STARTUPINFO()
startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
startupinfo.wShowWindow = 0  # SW_HIDE

def get_airfoil_params_with_flap_effect_naca(airfoil_name, Reynolds_numbers, deflection_angles, phases, output_dir="ht_deflections", alpha_sweep=(-15,25)):
    """
    Run XFOIL using built-in NACA airfoil with flap deflections at different Re.
    
    Args:
        airfoil_name: e.g., "2412"
        Reynolds_numbers: list of Re values
        deflection_angles: list of flap deflections in degrees
        phases: list of labels (e.g., flight phases) corresponding to Re
        output_dir: directory to save polar files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Path to XFOIL executable
    xfoil_path = os.path.join(os.getcwd(), r"..\XFOIL6.99\xfoil.exe")
    
    # Check if XFOIL exists
    if not os.path.isfile(xfoil_path):
        raise FileNotFoundError(f"XFOIL executable not found at: {xfoil_path}")

    for deflection in deflection_angles:
        for Re, phase in zip(Reynolds_numbers, phases):
            print(f"Running: NACA {airfoil_name}, Deflection={deflection}°, Re={Re} ({phase})")

            # Output polar file
            output_polar = os.path.join(output_dir, f"NACA{airfoil_name}_def ({deflection} deg)_{phase}.plr")
            
            # Build XFOIL command input
            commands = f"""
            NACA {airfoil_name}
            PANE
            GDES
            FLAP
            0.7
            0.0
            {deflection}
            EXEC
            
            PANE                    ! Critical: repanel after modification
            OPER
            VISC {Re}
            ITER 1000
            PACC
            {output_polar}
            
            ASEQ {alpha_sweep[0]} {alpha_sweep[1]} 0.5           ! Alpha sweep: -5 to +15 deg, 0.5 deg step
            QUIT
            """.strip()

            # Write commands to temp input file
            input_file = 'xfoil_temp_input.txt'
            with open(input_file, 'w') as f:
                f.write(commands)

            # print(commands)
            # return
            # Run XFOIL
            try:
                with open(input_file, 'r') as infile:
                    result = subprocess.run(
                        xfoil_path,
                        stdin=infile,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=60,  # seconds
                        creationflags = creationflags,
                        startupinfo=startupinfo
                    )
                
                # log_file = output_polar.replace(".plr", "_log.txt")
                # with open(log_file, 'w') as f:
                #     f.write(f"XFOIL Input:\n{commands}\n\n")
                #     # f.write(f"STDOUT:\n{result.stdout}\n\n")
                #     f.write(f"STDERR:\n{result.stderr}\n")
            except Exception as e:
                print(f"❌ Failed to run XFOIL: {e}")
                # pass

    # Cleanup
            if os.path.exists('xfoil_temp_input.txt'):
                os.remove('xfoil_temp_input.txt')

            # return