import matplotlib.pyplot as plt
from tides.parse_rt import parse_output, get_length
from tides.rt_spec import abs_spec

result = parse_output('H2_Ehrenfest.pyo')

HH_dist = get_length(result['coords'], [1,2])

plt.figure()
plt.plot(result['time'], HH_dist)
plt.ylabel(r'R(H-H) ($\mathrm{\AA}$)')
plt.xlabel('Time (au)')
plt.savefig('H2_Ehrenfest_Distance.png', bbox_inches='tight')

plt.figure()
plt.plot(result['time'], (result['energy'] - result['energy'][0])*27.2114)
plt.ylabel(r'$\Delta$Energy (eV)')
plt.xlabel('Time (au)')
plt.savefig('H2_Ehrenfest_Energy.png', bbox_inches='tight')
