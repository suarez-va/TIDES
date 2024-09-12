from pyscf import gto, dft, scf
import numpy as np
import rt_utils
import basis_utils
import rt_ehrenfest
from rt_cap import MOCAP
from rt_utils import get_noscf_orbitals
import rt_scf

dimer = gto.Mole()#basis = 'augccpvdz')
water1 = gto.Mole()#basis = 'augccpvdz')
water2 = gto.Mole()#basis = 'augccpvdz')
dimer.atom = '''
 O               -0.32314674    -1.47729686     0.00097471
 H               -1.27378046    -1.61942308     0.05721373
 H               -0.21933958    -0.51854906    -0.08219258
 H                0.76772643     1.70333069    -0.74899050
 O                0.30561043     1.34554668    -0.00175700
 H                0.78912388     1.57738619     0.78525838
'''
 #H                0.76772643     1.70333069    -0.74899050


water1.atom = '''
 O               -0.32314674    -1.47729686     0.00097471
 H               -1.27378046    -1.61942308     0.05721373
 H               -0.21933958    -0.51854906    -0.08219258
 H                0.76772643     1.70333069    -0.74899050
'''
water1.spin=1
water2.atom = '''
 O                0.30561043     1.34554668    -0.00175700
 H                0.78912388     1.57738619     0.78525838
'''
water2.spin=1
#H                0.76772643     1.70333069    -0.74899050
#'''

dimer.build()
water1.build()
water2.build()

dimer = scf.UHF(dimer)#.x2c()
water1 = scf.UHF(water1)#.x2c()
water2 = scf.UHF(water2)#.x2c()
#dimer = scf.UKS(dimer); dimer.xc = 'PBE0'
#water1 = scf.UKS(water1); water1.xc = 'PBE0'
#water2 = scf.UKS(water2); water2.xc = 'PBE0'
#dimer = scf.UKS(dimer); dimer.xc = 'B3LYP' 
#water1 = scf.UKS(water1); water1.xc = 'B3LYP' 
#water2 = scf.UKS(water2); water2.xc = 'B3LYP' 

dimer.kernel()
water1.kernel()
water2.kernel()

# Calculate noscf basis to print orbital occupations in
#noscf_orbitals = basis_utils.noscfbasis(dimer, water1, water2)
#noscf_orbitals = dimer.mo_coeff

rt_water = rt_ehrenfest.RT_Ehrenfest(dimer, 1, 250, filename='H2O', prop="magnus_interpol", frequency=1, verbose=9, Ne_step=1, N_step=1, get_mo_coeff_print=get_noscf_orbitals)

#rt_water = rt_scf.RT_SCF(dimer, 1, 10, filename='H2O', prop="magnus_interpol", frequency=1, chkfile='a', verbose=3)
# Declare which observables to be calculated/printed
rt_water.observables.update(energy=True, mo_occ=True, charge=True, nuclei=True)

# Create object for complex absorbing potential and add to rt object
CAP = MOCAP(0.5, 0.0477, 1.0, 10.0)#, dimer.get_ovlp())
rt_water.add_potential(CAP)

# Remove electron from 4th molecular orbital (in SCF basis)
# Input the two water fragments for their charge to be calculated
rt_utils.excite(rt_water, 4)
#rt_water._scf.mo_occ = rt_water.occ
rt_utils.input_fragments(rt_water, water1, water2)

# Start calculation, send in noscf_orbitals to print
rt_water.kernel()#mo_coeff_print=noscf_orbitals)#, match_indices_array = [[0,1,2],[3,4,5]])

