import numpy as np

'''
Real-time SCF Applied Potential
'''

class electric_field:
    def __init__(self, field_type=None, amplitude=[0,0,0], center=0, frequency=0, width=0, phase=0, spin='total'):

        # Some attributes are irrelevant depending on field type

        self.field_type = field_type
        self.amplitude = np.array(amplitude)
        self.center = center
        self.frequency = frequency
        self.width = width
        self.phase = phase
        self.spin = spin ############ Currently not in use

    def delta_energy(self):
        return self.amplitude

    def gaussian_energy(self, rt_mf):
        return self.amplitude * ((np.exp(-1 * (rt_mf.t - self.center) ** 2 / (2 * self.width ** 2))) * np.sin(self.width * rt_mf.t + self.phase))

    def hann_energy(self, rt_mf):
        return self.amplitude * ((np.sin(np.pi / self.width * (rt_mf.t - self.center - self.width / 2))) ** 2 * np.sin(self.width * rt_mf.t + self.phase))

    def resonant_energy(self, rt_mf):
        return self.amplitude * np.sin(self.width * rt_mf.t + self.phase)

    def calculate_field_energy(self, rt_mf):
        if self.field_type == 'delta' and rt_mf.t == self.center:
            return self.delta_energy()

        elif self.field_type == 'gaussian':
            return self.gaussian_energy(rt_mf)

        elif self.field_type == 'hann':
            return self.hann_energy(rt_mf)

        elif self.field_type == 'resonant':
            return self.resonant_energy(rt_mf)

        else:
            return np.array([0.0, 0.0, 0.0])

def applyfield(rt_mf, fock):
    efield_energy = np.array([0.0, 0.0, 0.0])        # x, y, and z components of efield

    for field in rt_mf.efield:
        efield_energy += field.calculate_field_energy(rt_mf)
    vapp = np.einsum('xij,x->ij', -1 * rt_mf._scf.mol.intor('int1e_r', comp=3), efield_energy)
    return fock + vapp

def static_bfield(rt_mf):
    x_bfield = rt_mf.bfield[0]
    y_bfield = rt_mf.bfield[1]
    z_bfield = rt_mf.bfield[2]

    hcore = rt_mf._scf.get_hcore()
    Nsp = int(rt_mf.ovlp.shape[0]/2)

    ovlp = rt_mf.ovlp[:Nsp,:Nsp]
    hcore = hcore[:Nsp,:Nsp]

    hprime = np.zeros([2*Nsp,2*Nsp], dtype=complex)

    hprime[:Nsp,:Nsp] = hcore + 0.5 * z_bfield * ovlp
    hprime[Nsp:,Nsp:] = hcore - 0.5 * z_bfield * ovlp
    hprime[Nsp:,:Nsp] = 0.5 * (x_bfield + 1j * y_bfield) * ovlp
    hprime[:Nsp,Nsp:] = 0.5 * (x_bfield - 1j * y_bfield) * ovlp
    rt_mf._scf.get_hcore = lambda *args: hprime
