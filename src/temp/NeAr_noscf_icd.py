from pyscf import gto, dft, scf
import numpy as np
import rt_scf
import rt_utils
import basis_utils
from rt_cap import MOCAP

NeAr_mol = gto.Mole()
Ar_mol = gto.Mole()
Ne_mol = gto.Mole()

NeAr_mol.atom = '''
   Ar    1   18      0.0000000000     0.0000000000     0.0000000000
   Ne    2   10      0.0000000000     0.0000000000     6.5762464403
'''
Ar_mol.atom = '''
   Ar    1   18      0.0000000000     0.0000000000     0.0000000000
'''
Ne_mol.atom = '''
   Ne    2   10      0.0000000000     0.0000000000     6.5762464403
'''

NeAr_mol.basis = 'ccpvdz'
Ne_mol.basis = 'ccpvdz'
Ar_mol.basis = 'ccpvdz'

NeAr_mol.build()
Ar_mol.build()
Ne_mol.build()

NeAr = scf.UHF(NeAr_mol)
Ar = scf.UHF(Ar_mol)
Ne = scf.UHF(Ne_mol)

NeAr.kernel()
Ar.kernel()
Ne.kernel()

# Calculate noscf basis to print orbital occupations in
noscf_orbitals = basis_utils.noscfbasis(NeAr, Ar, Ne)
NeAr.mo_coeff = noscf_orbitals
#basis_utils.print2molden(NeAr, 'NeAr_NOSCF', noscf_orbitals)
#basis_utils.print2molden(NeAr, 'NeAr_SCF')

rt_NeAr = rt_scf.RT_SCF(NeAr,1, 2500)
# Declare which observables to be calculated/printed
rt_NeAr.observables.update(energy=True, mo_occ=True, charge=True)

# Create object for complex absorbing potential and add to rt object
CAP = MOCAP(0.5, 0.0477, 1.0, 10.0)
rt_NeAr.add_potential(CAP)
# Input the two water fragments for their charge to be calculated
rt_utils.excite(rt_NeAr, 11)
rt_utils.input_fragments(rt_NeAr, Ar, Ne)

# Start calculation, send in noscf_orbitals to print
rt_NeAr.kernel(mo_coeff_print=noscf_orbitals)
