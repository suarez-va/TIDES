import numpy as np
from pyscf import gto, scf, dft
from pyscf.scf import hf ##
from rt_scf import RT_SCF
import rt_utils
from staticfield import static_bfield
from rt_utils import input_fragments
from hirshfeld import hirshfeld_partition

mag_z = 0.000085 # in au


LiTrimerMol = gto.M(
        verbose = 0,
        atom='''
 Li                 0.00000000    0.909326674 0.00000000
 Li                 -1.05    -0.909326674 0.00000000
 Li                 1.05    -0.909326674 0.00000000
''',
        basis='3-21G',
    spin = 1)

LiTrimer = scf.ghf.GHF(LiTrimerMol)
LiTrimer.kernel()


static_bfield(LiTrimer, [0,0,mag_z])

LiTrimer.kernel()
rt_mf = RT_SCF(LiTrimer, 1, 82000)
rt_mf.observables.update(dipole=True, mag=True, charge=True, atom_charges=True, hirshfeld_mags=True, hirshfeld_charges=True)

rt_mf.kernel()
