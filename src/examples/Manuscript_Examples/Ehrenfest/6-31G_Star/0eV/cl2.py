from pyscf import gto, dft, scf
import numpy as np
from tides import rt_ehrenfest, rt_utils, basis_utils

mol = gto.Mole()

mol.atom = '''
Cl           0 0 0
Cl           0 0 2.0
'''

mol.basis = '6-31G*'

mol.build()

cl2 = dft.UKS(mol)

cl2.xc = 'B3LYP'

cl2.kernel()

basis_utils.print2molden(cl2, filename='SCForbitals')
rt_cl2 = rt_ehrenfest.RT_Ehrenfest(cl2,0.5,10000,frequency=1, verbose=5, Ne_step=1, N_step=1)

# Declare which observables to be calculated/printed
rt_cl2.observables.update(dipole=True, charge=True, energy=True, atom_charges=True, nuclei=True, mo_occ=True, hirshfeld_charges=True, spin_square=True)

rt_cl2.kernel()
