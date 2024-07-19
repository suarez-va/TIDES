import numpy as np

'''
Real-time SCF Output Functions
'''

def update_output(rt_mf):
    rt_mf.log.note(f"Current Time (AU): {rt_mf.current_time:.8f} \n")
    for key, function in rt_mf.observables_functions.items():
        function[1](rt_mf)

    rt_mf.log.note(f"{'=' * 25} \n")

def print_energy(rt_mf):
    energy = rt_mf.energy[-1]
    rt_mf.log.note(f"Total Energy (AU): {energy[0]} \n")
    if len(energy) > 1:
        for index, fragment in enumerate(energy[1:]):
            rt_mf.log.note(f"Fragment {index + 1} Energy (AU): {fragment} \n")

def print_mo_occ(rt_mf):
    mo_occ = rt_mf.mo_occ[-1]
    rt_mf.log.note(f"Molecular Orbital Occupations: {mo_occ} \n")

def print_charge(rt_mf):
    charge = rt_mf.charge[-1]
    rt_mf.log.note(f"Total Electronic Charge: {np.real(charge[0])} \n")
    if len(charge) > 1:
        for index, fragment in enumerate(charge[1:]):
            rt_mf.log.note(f"Fragment {index + 1} Electronic Charge: {np.real(fragment)} \n")

def print_dipole(rt_mf):
    dipole = rt_mf.dipole[-1]
    rt_mf.log.note(f"Total Dipole Moment [X, Y, Z] (AU): {dipole[0]} \n")
    if len(dipole) > 1:
        for index, fragment in enumerate(dipole[1:]):
            rt_mf.log.note(f"Fragment {index + 1} Dipole Moment [X, Y, Z] (AU): {fragment} \n")

def print_mag(rt_mf):
    mag = rt_mf.mag[-1]
    rt_mf.log.note(f"Total Magnetization [X, Y, Z]: {np.real(mag[0])} \n")
    if len(mag) > 1:
        for index, fragment in enumerate(mag[1:]):
            rt_mf.log.note(f"Fragment {index + 1} Magnetization [X, Y, Z] (AU): {fragment} \n")
