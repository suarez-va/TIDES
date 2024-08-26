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

ehrenfest_nacl = rt_ehrenfest.RT_Ehrenfest(nacl, 0.5, 1250, filename="NaCl", prop=None, frequency=1, orth=None, chkfile=None, verbose=3, Ne_step=1, N_step=1)
ehrenfest_nacl.nuc.vel[1,1] = 1.8e-5
ehrenfest_nacl.observables.update(nuclei=True, energy=True)
ehrenfest_nacl.kernel()

