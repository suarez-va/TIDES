import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.patches as mpatches
from tides.parse_rt import parse_output, get_length
from MDAnalysis.coordinates.XYZ import XYZReader
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 0.75
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 0.75
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 0.5
mpl.rcParams['legend.facecolor'] = 'None'
mpl.rcParams['legend.edgecolor'] = 'k'


def get_results(filename):
    xyz = XYZReader(filename, dt=50)
    xyz.units['time'] = 'au'

    time = []
    positions = []
    for ts in xyz:
        time.append(ts.time)
        positions.append(np.array(ts.positions).astype(np.float64))
    time = np.array(time)
    positions = np.array(positions)

    dist = get_length(positions, [1,2])
    return time, dist


time0_G, _0ev_G = get_results('6-31G/0eV/trajectory.xyz')
time1_G, _1ev_G = get_results('6-31G/1eV/trajectory.xyz')
time2_G, _2ev_G = get_results('6-31G/2eV/trajectory.xyz')
time3_G, _3ev_G = get_results('6-31G/3eV/trajectory.xyz')

time0_Gs, _0ev_Gs = get_results('6-31G_Star/0eV/trajectory.xyz')
time1_Gs, _1ev_Gs = get_results('6-31G_Star/1eV/trajectory.xyz')
time2_Gs, _2ev_Gs = get_results('6-31G_Star/2eV/trajectory.xyz')
time3_Gs, _3ev_Gs = get_results('6-31G_Star/3eV/trajectory.xyz')

def Cl2_Dissociation():

    fig, axs = plt.subplots(2,1, figsize=(3.36, 2.52*2), dpi=600, sharex=True)
    plt.subplots_adjust(hspace=0.1)
    axs[0].grid(True)
    axs[0].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[0].plot(time0_G / 41.34, _0ev_G, 'C0', linewidth=1.5, label='0')
    axs[0].plot(time1_G / 41.34, _1ev_G, 'C1', linewidth=1.5, label='1')
    axs[0].plot(time2_G / 41.34, _2ev_G, 'C2', linewidth=1.5, label='2')
    axs[0].plot(time3_G / 41.34, _3ev_G, 'C3', linewidth=1.5, label='3')

    axs[0].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[0].xaxis.set_major_locator(MultipleLocator(40))

    axs[0].set_xlim([0,120])
    axs[0].set_ylim([1.5,3.99])
    axs[0].text(0.05, 0.9, 'a)', fontsize=12, va='center', transform=axs[0].transAxes)
    axs[0].text(0.03, .08, '6-31G', fontsize=15, transform=axs[0].transAxes)

    axs[1].grid(True)
    axs[1].tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    axs[1].plot(time0_Gs / 41.34, _0ev_Gs, 'C0', linewidth=1.5, label='0')
    axs[1].plot(time1_Gs / 41.34, _1ev_Gs, 'C1', linewidth=1.5, label='1')
    axs[1].plot(time2_Gs / 41.34, _2ev_Gs, 'C2', linewidth=1.5, label='2')
    axs[1].plot(time3_Gs / 41.34, _3ev_Gs, 'C3', linewidth=1.5, label='3')

    axs[1].yaxis.set_major_locator(MultipleLocator(0.5))
    axs[1].xaxis.set_major_locator(MultipleLocator(40))
    axs[1].set_xlabel('Time (fs)', fontsize=12)
    axs[1].set_xlim([0,120])
    axs[1].set_ylim([1.5,3.99])
    axs[1].text(0.05, 0.9, 'b)', fontsize=12, va='center', transform=axs[1].transAxes)
    axs[1].text(0.03, .08, r'6-31G*', fontsize=15, transform=axs[1].transAxes)
    fig.text(-0.03, 0.5, r'Distance ($\mathrm{\AA}$)', fontsize=12, va='center', rotation='vertical')

    plt.savefig('Cl2_Dissociation.png', bbox_inches='tight')

Cl2_Dissociation()
