import matplotlib.pyplot as plt
from tides.parse_rt import parse_output, get_length
from tides.rt_spec import abs_spec

result = parse_output('NaCl_Ehrenfest.pyo')

# We will only plot every 100 points to get clearer plots. The frequency of nuclei/force updates were each 10. So nuclei are updated every 10 electronic steps, forces are updated every 10 nuclei steps. Meaning the forces are updated every 100 electronic steps.

time, energy = result['time'][::100], result['energy'][::100]
HH_dist = get_length(result['coords'][::100,:,:], [1,2])

plt.figure()
plt.plot(time, HH_dist)
plt.ylabel(r'R(H-H) ($\mathrm{\AA}$)')
plt.xlabel('Time (au)')
plt.savefig('NaCl_Ehrenfest_Distance.png', bbox_inches='tight')

plt.figure()
plt.plot(time, (energy - energy[0])*27.2114)
plt.ylabel(r'$\Delta$Energy (eV)')
plt.xlabel('Time (au)')
plt.savefig('NaCl_Ehrenfest_Energy.png', bbox_inches='tight')
