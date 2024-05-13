import numpy as np
import scipy

'''
Real-time SCF molecular orbital complex absorbing potential (CAP)
'''

def mocap(rt_mf, fock):
    # Load CAP parameters
    expconst = rt_mf.CAP[0]
    emin = rt_mf.CAP[1]
    prefac = rt_mf.CAP[2]
    maxval = rt_mf.CAP[3]

    # Construct fock_orth without CAP
    fock_orth = np.dot(rt_mf.orth.T, np.dot(fock,rt_mf.orth))

    # Calculate MO energies
    mo_energy, mo_orth = scipy.linalg.eigh(fock_orth)
    mo_energy = np.real(mo_energy)

    # Construct damping terms
    damping_diagonal = []

    for energy in mo_energy:
        energy_corrected = energy - emin

        if energy_corrected > 0:
            damping_term = prefac * (1 - np.exp(expconst* energy_corrected))
            if damping_term < (-1 * maxval):
                damping_term = -1 * maxval
            damping_diagonal.append(damping_term)
        else:
            damping_diagonal.append(0)

    damping_diagonal = np.array(damping_diagonal).astype(np.complex128)

    # Construct damping matrix
    damping_matrix = np.diag(damping_diagonal)
    damping_matrix = np.dot(mo_orth, np.dot(damping_matrix,mo_orth.T))

    # Add damping matrix to fock
    fock_orth = fock_orth + 1j * damping_matrix

    return fock_orth