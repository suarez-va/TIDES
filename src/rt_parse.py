import numpy as np

'''
Real-time SCF Output File Parser
'''

def parse(rt_mf):
    '''
    Iterates through output file and assigns each
    collected observable vs. time as an attribute of the
    rt_mf object.
    '''
    with open(f"{rt_mf.filename}.txt", "r") as output_file:
      output_lines = output_file.readlines()

    time = parse_time(output_lines)

    if rt_mf.observables['energy']:
        energy = parse_energy(output_lines)
        rt_mf.energy = np.stack((time, energy)).T

    if rt_mf.observables['charge']:
        charge = parse_charge(output_lines)
        rt_mf.charge = np.stack((time, charge)).T

    if rt_mf.observables['dipole']:
        dipole = parse_dipole(output_lines)
        rt_mf.dipole = np.column_stack((time, dipole))

    if rt_mf.observables['mag']:
        mag = parse_mag(output_lines)
        rt_mf.mag = np.column_stack((time, mag))

    if rt_mf.observables['mo_occ']:
        mo_occ = parse_mo_occ(output_lines)
        rt_mf.mo_occ = np.column_stack((time, mo_occ))


def parse_time(output_lines):
    time = []
    for line in output_lines:
        if 'Current Time (AU):' in line:
            time.append(float(line.split(':')[1]))
    return np.array(time)

def parse_energy(output_lines):
    energy = []
    for line in output_lines:
        if 'Total Energy (AU):' in line:
            energy.append(float(line.split(':')[1]))
    return np.array(energy)

def parse_charge(output_lines):
    charge = []
    for line in output_lines:
        if 'Electronic Charge (Total):' in line:
            charge.append(float(line.split(':')[1]))
    return np.array(charge)

def parse_dipole(output_lines):
    dipole = []
    for line in output_lines:
        if 'Dipole Moment [X, Y, Z] (AU):' in line:
            dipole.append(np.array(line.split(':')[1].strip()[1:-1].split()).astype(float))
    return np.array(dipole)

def parse_mag(output_lines):
    mag = []
    for line in output_lines:
        if 'Magnetization [X, Y, Z]' in line:
            mag.append(np.array(line.split(':')[1].strip()[1:-1].split()).astype(float))
    return np.array(mag)

def parse_mo_occ(output_lines):
    mo_occ = []
    for index, line in enumerate(output_lines):
        if 'Molecular Orbital Occupations:' in line:
            all_mo_found = False
            moiter = 1
            extended_line = line
            while not all_mo_found:
                if '=' not in output_lines[index + moiter]:
                    extended_line = " ".join((extended_line, output_lines[index + moiter]))
                else:
                    all_mo_found = True
                moiter += 1
            mo_occ.append(np.array(extended_line.split(':')[1].strip()[1:-1].split()).astype(float))
    return np.array(mo_occ)
