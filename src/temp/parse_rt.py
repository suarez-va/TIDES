import numpy as np
import matplotlib.pyplot as plt

def parse_output(filename, mol_length=0):
    with open(filename, 'r') as f:
        lines = f.readlines()
    time = []
    energy = []
    dipole = []
    mo_occ = []
    charge = []
    mag = []
    mol_length = mol_length
    coords = []
    for idx, line in enumerate(lines):
        if 'Current Time' in line:
            time.append(get_time(line))
        if 'Total Dipole Moment' in line:
            dipole.append(get_dipole(line))
        if 'Total Energy' in line:
            energy.append(get_energy(line))
        if 'Molecular Orbital Occupations' in line:
            mo_occ.append(get_mo_occ(line))
        if 'Total Electronic Charge' in line:
            charge.append(get_charge(line))
        if 'Total Magnetization' in line:
            mag.append(get_mag(line))
        if 'Nuclear Coordinates' in line:
            coords.append(get_coords(lines[idx+1:idx+mol_length+1]))

    time = np.array(time)
    energy = np.array(energy)
    dipole = np.array(dipole)
    mo_occ = np.array(mo_occ)
    charge = np.array(charge)
    mag = np.array(mag)
    coords = np.array(coords)
    return time, energy, dipole, mo_occ, charge, mag, coords

def plot_dip(time, dipole, name):
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time / 41.34, dipole)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Dipole Moment (au)', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_mag(time, mag, name):
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time / 41.34, mag)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Magnetization (au)', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_charge(time, charge, name):
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time / 41.34, charge)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Electronic Charge', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_energy(time, energy, name):
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time / 41.34, energy)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Energy (au)', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_lens(time, energy, name):
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time / 41.34, energy)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Distance (A)', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_mo_occ(time, mo_occ, name, mo_lim=10):
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    for i in range(mo_lim):
        plt.plot(time / 41.34, mo_occ[:, i], label=i)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Occupation', fontsize=15)
    #plt.ylim([-0.1,2.1])
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def get_time(line):
    return float(line.split()[3])

def get_energy(line):
    return float(line.split()[3])

def get_dipole(line):
    x = float(line.split()[-3])
    y = float(line.split()[-2])
    z = float(line.split()[-1])
    return [x, y, z]

def get_mag(line):
    x = float(line.split()[-3])
    y = float(line.split()[-2])
    z = float(line.split()[-1])
    return [x, y, z]

def get_mo_occ(line):
    return np.array(line.split('Molecular Orbital Occupations: ')[1].split()).astype(np.float64)

def get_charge(line):
    return float(line.split()[3])

def get_coords(lines):
    coords = []
    for line in lines:
        mol_coords = []
        for i in range(1,4):
            mol_coords.append(float(line.split()[i]))
        coords.append(mol_coords)
    return coords

def get_length(coords, atoms):
    # Gets length between 2 atoms
    atoms = [atoms[i] - 1 for i, _ in enumerate(atoms)]
    lens = []
    for coord in coords:
        dist_3d = coord[atoms[1]] - coord[atoms[0]]
        dx, dy, dz = dist_3d[0], dist_3d[1], dist_3d[2]
        lens.append(np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    return lens

def main():
    
    filename = 'NaCl.txt'
    #filename = 'b.out'
    time, energy, dipole, mo_occ, charge, mag, coords = parse_output(filename, mol_length=2)
    
    #OOlen = get_length(coords, [1,4])
    #OHlen = get_length(coords, [1,5])
    nacllen = get_length(coords, [1,2])
    plot_energy(time, energy, 'en.png')
    plot_lens(time, nacllen, 'naccllen.png')
    #plot_energy(time, energy, 'en.png')
    #plot_lens(time, OOlen, 'OOlen.png')
    #plot_lens(time, OHlen, 'OHlen.png')
    #plot_mo_occ(time, mo_occ, 'moc.png', mo_lim=15)
    #plot_mag(time, mag[:,0], 'magx.png')
    #plot_mag(time, mag[:,1], 'magy.png')
    #plot_mag(time, mag[:,2], 'magz.png')
    #plot_charge(time, charge, 'charge.png')

if __name__ == '__main__':
    main()
