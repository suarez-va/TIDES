import numpy as np

'''
SCF Time-Independent Applied Potential
'''

def static_bfield(mf, bfield):
    x_bfield = bfield[0]
    y_bfield = bfield[1]
    z_bfield = bfield[2]

    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    Nsp = int(ovlp.shape[0]/2)

    ovlp = ovlp[:Nsp,:Nsp]
    hcore = hcore[:Nsp,:Nsp]

    hprime = np.zeros([2*Nsp,2*Nsp], dtype=complex)

    hprime[:Nsp,:Nsp] = hcore + 0.5 * z_bfield * ovlp
    hprime[Nsp:,Nsp:] = hcore - 0.5 * z_bfield * ovlp
    hprime[Nsp:,:Nsp] = 0.5 * (x_bfield + 1j * y_bfield) * ovlp
    hprime[:Nsp,Nsp:] = 0.5 * (x_bfield - 1j * y_bfield) * ovlp
    mf.get_hcore = lambda *args: hprime
