from pyscf import gto, dft, scf
import numpy as np
import rt_scf
import rt_utils
import basis_utils
from rt_cap import MOCAP

hcn = gto.Mole()
hcn.atom = '''
H 0 0 0
C 0 0 2.03095373
N 0 0 4.21747158
'''
hcn.basis='3-21G'
hcn.unit = 'B'
hcn.build()

hcn = scf.RHF(hcn)
hcn.kernel()

rt_hcn = rt_scf.RT_SCF(hcn,1, 1)
rt_hcn.observables.update(energy=True, mo_occ=True, charge=True, atom_charges=True, hirshfeld_charges=True)
rt_hcn.kernel()
