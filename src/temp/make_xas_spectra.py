import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from parse_rt import parse_output
from rt_spec import abs_spec

filename = 'fecl.out'
time, energy, dipole, mo_occ, charge, mag, coords = parse_output(filename)

#abs_spec(time, dipole, 'osc_str', kick_str=0.0001, pad=50000, damp=50, preprocess_zero=True)
abs_spec(time, dipole, 'osc_str', kick_str=0.0001, damp=0, preprocess_zero=True)

a = np.loadtxt('osc_str.txt')

freq = a[:,0] * 27.21
osc_str = a[:,1]

plt.plot(freq, osc_str)
plt.xlim([7000,7200])
#plt.ylim([0,0.1])
plt.savefig('spec.png')
