import numpy as np
from pyscf import gto, dft, scf
import rt_scf_prop
import rt_observables
import rt_output

'''
Real-time SCF Main Driver
'''

class RT_SCF:
    def __init__(self, mf, timestep, frequency, total_steps,
                filename, prop=None, orth=None):

        self.timestep = timestep
        self.frequency = frequency
        self.total_steps = total_steps
        self.filename = filename
        self._scf = mf
        self.ovlp = self._scf.get_ovlp()  # assumes constant basis
        self.occ = self._scf.get_occ()
        self.fragments = []
        self.potential = []
        if prop is None: self.prop = "magnus_interpol"
        if orth is None: self.orth = scf.addons.canonical_orth_(self.ovlp)
        if mf.istype('UKS') | mf.istype('UHF'):
            self.nmat = 2
        else:
            self.nmat = 1

        self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)
        self.t = 0
        rt_observables.init_observables(self)

    def get_fock_orth(self, den_ao):
        self.fock = self._scf.get_fock(dm=den_ao)
        if self.potential: self.applypotential()
        return np.matmul(self.orth.T, np.matmul(self.fock, self.orth))

    def add_potential(self, *args):
        for v_ext in args:
            self.potential.append(v_ext)

    def applypotential(self):
        for v_ext in self.potential:
            self.fock = np.add(self.fock, v_ext.calculate_potential(self))

    def kernel(self, mo_coeff_print=None):
        rt_output.create_output_file(self)
        if mo_coeff_print is None: mo_coeff_print = self._scf.mo_coeff
        rt_scf_prop.propagate(self, mo_coeff_print)
        return self
