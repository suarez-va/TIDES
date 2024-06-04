import numpy as np

'''
Real-time SCF output file
'''

def create_output_file(rt_mf):
    with open(f"{rt_mf.filename}.txt", "w") as output_main:
        output_main.write("TEXT FOR BEGINNING OF OUTPUT FILE \n")
        output_main.write("=== Start Propagation === \n")
        output_main.write(f"{'=' * 25} \n")

def update_output_file(rt_mf):
    with open(f"{rt_mf.filename}.txt", "a") as output_main:
        output_main.write(f"Current Time (AU): {rt_mf.t:.8f} \n")
        for key, function in rt_mf.observables_functions.items():
            function[1](output_main, rt_mf.observables_values[key])

        output_main.write(f"{'=' * 25} \n")

def print_energy(output_main, energy):
    output_main.write(f"Total Energy (AU): {energy} \n")

def print_mo_occ(output_main, den_mo):
    output_main.write(f"Molecular Orbital Occupations: {np.diagonal(den_mo)} \n")

def print_charge(output_main, charge):
    output_main.write(f"Electronic Charge (Total): {np.real(charge[0])} \n")
    if len(charge) > 1:
        for index, fragment in enumerate(charge[1:]):
            output_main.write(f"Electronic Charge (Fragment {index + 1}): {np.real(fragment)} \n")

def print_dipole(output_main, dipole):
    output_main.write(f"Dipole Moment [X, Y, Z] (AU): {dipole} \n")

def print_mag(output_main, mag):
    output_main.write(f"Magnetization [X, Y, Z]: {np.real(mag)} \n")
