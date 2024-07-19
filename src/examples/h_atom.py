import numpy as np
from pyscf import gto, scf, dft
import rt_scf
import rt_vapp
from staticfield import static_bfield

mag_z = 0.000085 # in au

mol = gto.M(
	verbose = 0,
	atom='H 0 0 0',
	basis='STO-3G',
    spin = 1)

mf = scf.ghf.GHF(mol)
mf.kernel()

static_bfield(mf, [0,0,mag_z])
rt_mf = rt_scf.RT_SCF(mf, 1, 1, 10000)
rt_mf.observables.update(mag=True)

rt_mf.kernel()
