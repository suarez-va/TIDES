from pyscf import gto, scf, dft
import numpy as np
import rt_scf
import rt_utils
import basis_utils

dimer = gto.Mole()
Li1 = gto.Mole()
Li2 = gto.Mole()

dimer.atom = '''
Li 0.0 0.0 0.0
Li 0.0 0.0 5.0
'''

Li1.atom = '''
Li 0.0 0.0 0.0
'''

Li2.atom = '''
Li 0.0 0.0 5.0
'''

dimer.basis = 'aug-cc-pvqz'
Li1.basis = 'aug-cc-pvqz'
Li2.basis = 'aug-cc-pvqz'

dimer.charge = +1
dimer.spin = 1
dimer.build()

Li1.charge = +1
Li1.spin = 0
Li2.charge = 0
Li2.spin = 1
Li1.build()
Li2.build()

dimer = scf.UHF(dimer)
Li1 = scf.UHF(Li1)
Li2 = scf.UHF(Li2)


dimer.kernel()
Li1.kernel()
Li2.kernel()

dimer.mo_coeff = basis_utils.noscfbasis(dimer,Li1,Li2)
rt_mf = rt_scf.rt_scf(dimer,1,1,200,"Li")
rt_utils.input_fragments(rt_mf,range(0,1),range(1,2))


rt_mf.kernel(mo_coeff_print=noscf_orbitals)
