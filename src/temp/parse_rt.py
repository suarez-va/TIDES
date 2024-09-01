import numpy as np
import matplotlib.pyplot as plt

def parse_output(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    time = []
    energy = []
    dipole = []
    mo_occ = []
    charge = []
    mag = []
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

    time = np.array(time)
    energy = np.array(energy)
    dipole = np.array(dipole)
    mo_occ = np.array(mo_occ)
    charge = np.array(charge)
    mag = np.array(mag)
    return time, energy, dipole, mo_occ, charge, mag

def plot_dip(time, dipole, name):
    time /= 41.34
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time, dipole)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Dipole Moment (au)', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_mag(time, mag, name):
    time /= 41.34
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time, mag)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Magnetization (au)', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_charge(time, charge, name):
    time /= 41.34
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time, charge)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Electronic Charge', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_energy(time, energy, name):
    time /= 41.34
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    plt.plot(time, energy)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Energy (au)', fontsize=15)
    plt.savefig(name, bbox_inches='tight')

def plot_mo_occ(time, mo_occ, name, mo_lim=10):
    time /= 41.34
    plt.figure()
    ax = plt.axes()
    #ax.axis([0, 200 / 41.34, -0.5, 0.5])
    ax.tick_params(which='both',direction='in', top=True, right=True)
    for i in range(mo_lim):
        plt.plot(time, mo_occ[:, i], label=i)
    plt.xlabel('Time(fs)', fontsize=15)
    plt.ylabel('Occupation', fontsize=15)
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

def main():
    
    filename = 'NaCl.txt'
    time, energy, dipole, mo_occ, charge, mag = parse_output(filename)
    
    plot_energy(time, energy, 'en.png')

    #plot_mo_occ(time, mo_occ, 'moc.png', mo_lim=10)
    #plot_mag(time, mag[:,0], 'magx.png')
    #plot_mag(time, mag[:,1], 'magy.png')
    #plot_mag(time, mag[:,2], 'magz.png')
    #plot_charge(time, charge, 'charge.png')

if __name__ == '__main__':
    main()
