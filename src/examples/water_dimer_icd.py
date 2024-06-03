from pyscf import gto, dft, scf
import numpy as np
import rt_scf
import rt_utils
import basis_utils
from rt_cap import mocap

dimer = gto.Mole()
water1 = gto.Mole()
water2 = gto.Mole()

dimer.atom = '''
 O                  1.49137509   -0.00861627    0.00000100
 H                  0.53614403    0.12578597    0.00000098
 H                  1.86255228    0.87394752    0.00000101
 O                  -1.41904607    0.10858435    0.00000006
 H                  -1.76001996   -0.36619300   -0.76029109
 H                  -1.76001996   -0.36619300    0.76029121
'''
water1.atom = '''
 O                  1.49137509   -0.00861627    0.00000100
 H                  0.53614403    0.12578597    0.00000098
 H                  1.86255228    0.87394752    0.00000101
'''
water2.atom = '''
 O                  -1.41904607    0.10858435    0.00000006
 H                  -1.76001996   -0.36619300   -0.76029109
 H                  -1.76001996   -0.36619300    0.76029121
'''

dimer.build()
water1.build()
water2.build()

dimer = scf.UHF(dimer)
water1 = scf.UHF(water1)
water2 = scf.UHF(water2)

dimer.kernel()
water1.kernel()
water2.kernel()

noscf_orbitals = basis_utils.noscfbasis(dimer, water1, water2)

rt_water = rt_scf.rt_scf(dimer,1,1,1500,"H2OH2O")
rt_water.prop = 'magnus_step'
CAP = mocap(0.5, 0.0477, 1.0, 10.0, dimer.get_ovlp())
rt_water.add_potential(CAP)

rt_utils.excite(rt_water, 4)
rt_utils.input_fragments(rt_water, range(0,3),range(3,6))


rt_water.kernel(mo_coeff_print=noscf_orbitals)
