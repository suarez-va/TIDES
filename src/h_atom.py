import numpy as np
from pyscf import gto, scf
import rt_ghf

### initialize variables
timestep = 0.05
steps = 50000
total_steps = 2080000
filename = "h_atom"


mag_z = 0.000085 # in au

### initialize static calculation
mol = gto.M(        
	verbose = 0,       
	atom='H 0 0 0',        
	basis='STO-3G',
    spin = 1)

### calculation to get initial hamiltonian
mf = scf.ghf.GHF(mol)
mf.scf()

### initialize hamiltonian

ovlp = mf.get_ovlp()
hcore = mf.get_hcore()

Nsp = int(ovlp.shape[0]/2)

ovlp = ovlp[:Nsp,:Nsp]
hcore = hcore[:Nsp,:Nsp]

hprime = np.zeros([2*Nsp,2*Nsp], dtype=complex)

hprime[:Nsp,:Nsp]= hcore + 0.5*mag_z*ovlp
hprime[Nsp:,Nsp:]= hcore - 0.5*mag_z*ovlp

### call dynamics 

mf.get_hcore = lambda *args: hprime

var = rt_ghf.GHF(mf, timestep, steps, total_steps, filename)

var.dynamics()

var.plot_mag()

var.plot_energy()

