import numpy as np
from pyscf import gto, scf, dft
import rt_scf
import rt_vapp

mag_z = 0.000085 # in au

mol = gto.M(
	verbose = 0,
	atom='H 0 0 0',
	basis='STO-3G',
    spin = 1)

mf = scf.ghf.GHF(mol)
mf.kernel()

rt_mf = rt_scf.rt_scf(mf, 0.05, 50000, 2080000, 'h_atom')

rt_mf.bfield = [0,0, mag_z]

rt_mf.prop = 'magnus_step'

rt_mf.kernel()
