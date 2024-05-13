import numpy as np
from pyscf import gto, scf
import rt_ghf

### initialize variables
timestep = 0.05       
#steps = 29000        
#total_steps = 2893870
steps = 33600
total_steps = 1680000
filename = "li_atom"

mag = -0.000085 # in au

mag_x = mag * np.sin(np.pi/4) * np.cos(np.pi/4)
mag_y = mag * np.sin(np.pi/4) * np.sin(np.pi/4)
mag_z = mag * np.cos(np.pi/4) 

### initialize static calculation
print("Initializing system.")

mol = gto.M(        
	verbose = 0,       
	atom='Li 0 0 0',        
    basis='3-21G',
    #basis='STO-3G',
    spin = 1)

### calculation to get initial hamiltonian
mf = scf.ghf.GHF(mol)
mf.scf()

### testing, delete later

den = mf.make_rdm1()
ovlp = mf.get_ovlp()
Nsp = int(ovlp.shape[0]/2)

den = mf.make_rdm1()

## initialize hamiltonian
print("Initializing Hamiltonian")

ovlp = mf.get_ovlp()
hcore = mf.get_hcore()

Nsp = int(ovlp.shape[0]/2)

ovlp = ovlp[:Nsp,:Nsp]
hcore = hcore[:Nsp,:Nsp]

hprime = np.zeros([2*Nsp,2*Nsp], dtype=complex)

hprime[:Nsp,:Nsp] = hcore + 0.5*mag_z*ovlp
hprime[Nsp:,Nsp:] = hcore - 0.5*mag_z*ovlp
hprime[:Nsp,Nsp:] = 0.5*(mag_x - 1j*mag_y)*ovlp
hprime[Nsp:,:Nsp] = 0.5*(mag_x + 1j*mag_y)*ovlp

### call dynamics 
print("Beginning dynamics.")

mf.get_hcore = lambda *args: hprime

var = rt_ghf.GHF(mf, timestep, steps, total_steps, filename)

var.dynamics()

var.plot_mag()

var.plot_energy()

print("Simulation completed successfully.")
