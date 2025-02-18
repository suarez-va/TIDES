import matplotlib.pyplot as plt
from tides.parse_rt import parse_output

result = parse_output('H_X2C.pyo')

plt.figure()
plt.plot(result['time'], result['energy']*27.2114)
plt.ylabel('Energy (eV)')
plt.xlabel('Time (au)')
plt.savefig('H_X2C_Energy.png', bbox_inches='tight')
