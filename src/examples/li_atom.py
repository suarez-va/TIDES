import numpy as np
from pyscf import gto, scf, dft
from rt_scf import RT_SCF
from staticfield import static_bfield
from rt_utils import input_fragments

mag = -0.000085 # in au

mag_x = mag * np.sin(np.pi/4) * np.cos(np.pi/4)
mag_y = mag * np.sin(np.pi/4) * np.sin(np.pi/4)
mag_z = mag * np.cos(np.pi/4) 

LiMol = gto.M(
	verbose = 0,
	atom='''
 Li                 0.00000000    0.00000000 0.00000000
''',
	basis='3-21G',
    spin = 1)


Li = scf.ghf.GHF(LiMol)

Li.kernel()

static_bfield(Li, [mag_x,mag_y,mag_z])
rt_mf = RT_SCF(Li, 1, 10000)

rt_mf.observables.update(dipole=True, mag=True)
rt_mf.kernel()

