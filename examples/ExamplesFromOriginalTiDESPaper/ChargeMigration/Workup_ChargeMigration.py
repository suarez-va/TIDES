import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.patches as mpatches
from tides.parse_rt import parse_output
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

def EthyleneStack():
    filename = 'ethylene_x4.out'
    result = parse_output(filename)
    time = result['time']
    hirsh_charges = result['hirsh_charge']
    a_charge = hirsh_charges[:,:6].sum(axis=1)
    b_charge = hirsh_charges[:,6:12].sum(axis=1)
    c_charge = hirsh_charges[:,12:18].sum(axis=1)
    d_charge = hirsh_charges[:,18:].sum(axis=1)

    plt.figure(figsize=(3.36, 2.52), dpi=600)
    ax = plt.axes()

    ax.tick_params(labelsize=8, which='both',direction='in', top=True, right=True)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    plt.plot(time / 41.34, 16 - a_charge, 'purple', linewidth=1.5, label='A')
    plt.plot(time / 41.34, 16 - b_charge, 'blue', linewidth=1.5, label='B')
    plt.plot(time / 41.34, 16 - c_charge, 'green', linewidth=1.5, label='C')
    plt.plot(time / 41.34, 16 - d_charge, 'red', linewidth=1.5, label='D')
    plt.xlim([-0.5,10.5])
    plt.ylim([-0.01, 1.1])
    plt.ylabel('Charge', fontsize=12)
    plt.xlabel('Time (fs)', fontsize=12)
    plt.text(0.05, 0.9, 'b)', fontsize=12, va='center', transform=ax.transAxes)

    plt.savefig('ethylene_x4.png', bbox_inches='tight')


EthyleneStack()
