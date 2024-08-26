import numpy as np

'''
Real-time Output Functions
'''

def update_output(rt_mf):
    rt_mf._log.note(f"Current Time (AU): {rt_mf.current_time:.8f} \n")
    for key, function in rt_mf._observables_functions.items():
        function[1](rt_mf)

    rt_mf._log.note(f"{'=' * 25} \n")

def _print_energy(rt_mf):
    energy = rt_mf._energy
    rt_mf._log.note(f"Total Energy (AU): {energy[0]} \n")
    if len(energy) > 1:
        for index, fragment in enumerate(energy[1:]):
            rt_mf._log.note(f"Fragment {index + 1} Energy (AU): {fragment} \n")

def _print_mo_occ(rt_mf):
    mo_occ = rt_mf._mo_occ
    rt_mf._log.note(f"Molecular Orbital Occupations: {' '.join(map(str,mo_occ))} \n")

def _print_charge(rt_mf):
    charge = rt_mf._charge
    rt_mf._log.note(f"Total Electronic Charge: {np.real(charge[0])} \n")
    if len(charge) > 1:
        for index, fragment in enumerate(charge[1:]):
            rt_mf._log.note(f"Fragment {index + 1} Electronic Charge: {np.real(fragment)} \n")

def _print_dipole(rt_mf):
    dipole = rt_mf._dipole
    rt_mf._log.note(f"Total Dipole Moment [X, Y, Z] (AU): {' '.join(map(str,dipole[0]))} \n")
    if len(dipole) > 1:
        for index, fragment in enumerate(dipole[1:]):
            rt_mf._log.note(f"Fragment {index + 1} Dipole Moment [X, Y, Z] (AU): {' '.join(map(str,fragment))} \n")

def _print_mag(rt_mf):
    mag = rt_mf._mag
    rt_mf._log.note(f"Total Magnetization [X, Y, Z]: {' '.join(map(str,np.real(mag[0])))} \n")
    if len(mag) > 1:
        for index, fragment in enumerate(mag[1:]):
            rt_mf._log.note(f"Fragment {index + 1} Magnetization [X, Y, Z] (AU): {' '.join(map(str,fragment))} \n")

def _print_nuclei(rt_mf):
    nuclei = rt_mf._nuclei
    rt_mf._log.note(f"Nuclear Coordinates (AU):")
    for index, label in enumerate(nuclei[0][0]):
        rt_mf._log.note(f" {label} {' '.join(map(lambda x: f'{x:.11f}' if x<0 else f' {x:.11f}', nuclei[1][index]))}")
    rt_mf._log.note(" ")

