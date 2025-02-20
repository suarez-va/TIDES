from tides.parse_rt import parse_output
import matplotlib.pyplot as plt

result = parse_output('LiTrimer_BField.pyo')
time, atom_mag = result['time'], result['hirsh_mag']


fig, axs = plt.subplots(3,1, figsize=(6.4, 4.8*3), sharex=True)
plt.subplots_adjust(hspace=0.1)
axs[0].plot(time / 41340, atom_mag[:,0,0], color=[1,0,0])
axs[0].plot(time / 41340, atom_mag[:,1,0], color=[0,0.75,0])
axs[0].plot(time / 41340, atom_mag[:,2,0], color=[0,0.5,0.75])

axs[0].set_ylabel(r'$\mathrm{M_x}$ (au)', fontsize=20)
axs[0].set_xlim([0,2])
axs[0].set_ylim([-0.25,0.25])

axs[1].plot(time / 41340, atom_mag[:,0,1], color=[1,0,0])
axs[1].plot(time / 41340, atom_mag[:,1,1], color=[0,0.75,0])
axs[1].plot(time / 41340, atom_mag[:,2,1], color=[0,0.5,0.75])
# For the y magnetization (the orthogonal plane) the magnetization stays at ~0.
axs[1].set_ylabel(r'$\mathrm{M_y}$ (au)', fontsize=20)
axs[1].set_xlim([0,2])
axs[1].set_ylim([-0.25,0.25])

axs[2].plot(time / 41340, atom_mag[:,0,2], color=[1,0,0])
axs[2].plot(time / 41340, atom_mag[:,1,2], color=[0,0.75,0])
axs[2].plot(time / 41340, atom_mag[:,2,2], color=[0,0.5,0.75])

axs[2].set_xlabel('Time (ps)', fontsize=20)
axs[2].set_ylabel(r'$\mathrm{M_z}$ (au)', fontsize=20)
axs[2].set_xlim([0,2])
axs[2].set_ylim([-0.25,0.25])

plt.savefig('LiTrimer_BField_Mag.png', bbox_inches='tight')
