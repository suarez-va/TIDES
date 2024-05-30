import numpy as np

'''
Real-time SCF molecular orbital complex absorbing potential (CAP)
'''


class mocap:
    def __init__(self, expconst, emin, prefac, maxval, ovlp, thr=1e-7):
        self.expconst = expconst
        self.emin = emin
        self.prefac = prefac
        self.maxval = maxval

        normlz = np.power(np.diag(ovlp), -0.5)
        Snorm = np.dot(np.diag(normlz), np.dot(ovlp, np.diag(normlz)))
        Sval, Svec = np.linalg.eigh(Snorm)

        self.y_orth = Svec[:,Sval>=thr] * np.sqrt(Sval[Sval>=thr])
        # Y = Xs^1/2 is needed to rotate damping matrix back to AO basis
        self.y_orth = np.dot(np.diag(normlz), self.y_orth)

    def calculate_potential(self, rt_mf):
        if len(rt_mf.dim) == 1:
            return self.calc_cap(rt_mf, rt_mf.fock[0])
        else:
            return np.stack((self.calc_cap(rt_mf, rt_mf.fock[0]), self.calc_cap(rt_mf, rt_mf.fock[1])))

    def calc_cap(self, rt_mf, fock):
        # Construct fock_orth without CAP
        fock_orth = np.dot(rt_mf.orth.T, np.dot(fock,rt_mf.orth))

        # Calculate MO energies
        mo_energy, mo_orth = np.linalg.eigh(fock_orth)
        mo_energy = np.real(mo_energy)

        # Construct damping terms
        damping_diagonal = []

        for energy in mo_energy:
            energy_corrected = energy - self.emin

            if energy_corrected > 0:
                damping_term = self.prefac * (1 - np.exp(self.expconst* energy_corrected))
                if damping_term < (-1 * self.maxval):
                    damping_term = -1 * self.maxval
                damping_diagonal.append(damping_term)
            else:
                damping_diagonal.append(0)

        damping_diagonal = np.array(damping_diagonal).astype(np.complex128)

        # Construct damping matrix
        damping_matrix = np.diag(damping_diagonal)
        damping_matrix = np.dot(mo_orth, np.dot(damping_matrix,mo_orth.T))

        # Rotate back to ao basis
        damping_matrix_ao = np.dot(self.y_orth, np.dot(damping_matrix, self.y_orth.T))
        return 1j * damping_matrix_ao
