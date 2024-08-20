from pyscf import gto, dft, scf
import numpy as np
import rt_ehrenfest

nacl = gto.Mole(basis = '321g')

nacl.atom = '''
 Na     0.00000000     0.00000000     0.00000000
 Cl     0.00000000     2.42100000     0.00000000
'''

nacl.build()

nacl = scf.RHF(nacl)

nacl.kernel()

ehrenfest_nacl = rt_ehrenfest.RT_EHRENFEST(nacl, 0.5, 25, 2500, filename="NaCl", prop=None, orth=None, chkfile=None, verbose=3, Ne_step=1, N_step=1)
ehrenfest_nacl.nuc.vel[1,1] = 1.8e-5
ehrenfest_nacl.kernel()

