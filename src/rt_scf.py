import numpy as np
from pyscf import gto, dft, scf
import rt_integrators
import rt_observables
import rt_output


'''
Real-time SCF main driver
'''

class rt_scf:
    def __init__(self, mf, timestep, frequency, total_steps, filename, prop=None, orth=None):
        self.timestep = timestep
        self.frequency = frequency
        self.total_steps = total_steps
        self.filename = filename
        self._scf = mf
        self.mag = False  # temp
        self.fragments = []
        self.ovlp = self._scf.get_ovlp()  # assumes constant basis
        self.potential = []
        self.occ = self._scf.get_occ()
        self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)
        self.t = 0

        if prop is None: self.prop = "magnus_interpol"
        if orth is None: self.orth = scf.addons.canonical_orth_(self.ovlp)

        if mf.istype('RKS') | mf.istype('RHF'):
            self.nmat = 1
        elif mf.istype('UKS') | mf.istype('UHF'):
            self.nmat = 2
        elif mf.istype('GKS') | mf.istype('GHF'):
            self.mag = True
            self.nmat = 1
        else:
            raise Exception('unknown scf method')

    def get_fock_orth(self, den_ao):
        self.fock = self._scf.get_fock(dm=den_ao)
        if self.potential: self.applypotential()
        return np.matmul(self.orth.T, np.matmul(self.fock, self.orth))

    def add_potential(self, *args):
        for v_external in args:
            self.potential.append(v_external)

    def applypotential(self):
        for v_external in self.potential:
            self.fock = np.add(self.fock, v_external.calculate_potential(self))

    def kernel(self, mo_coeff_print=None):
        rt_output.create_output_file(self)
        if mo_coeff_print is None: mo_coeff_print = self._scf.mo_coeff
        rt_integrators.propagate(self, mo_coeff_print)
