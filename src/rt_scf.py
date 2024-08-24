import numpy as np
from scipy.linalg import inv
from pyscf import gto, dft, scf
from pyscf.lib import logger
import rt_scf_prop
import rt_observables
import rt_output
from rt_utils import restart_from_chkfile
import os


'''
Real-time SCF Class
'''

class RT_SCF:
    def __init__(self, mf, timestep, frequency, total_steps,
                filename=None, prop=None, orth=None, chkfile=None, verbose=3):

        self.timestep = timestep
        self.frequency = frequency
        self.total_steps = total_steps
        self._scf = mf
        self.ovlp = self._scf.get_ovlp()
        self.occ = self._scf.get_occ()

        self.verbose = verbose
        self.potential = []
        self.fragments = {}

        if prop is None: prop = "magnus_interpol"
        if orth is None: orth = scf.addons.canonical_orth_(self.ovlp)
        self.prop = prop
        self.orth = orth

        if len(np.shape(self._scf.mo_coeff)) == 3:
            self.nmat = 2
        else:
            self.nmat = 1

        if filename is None:
            self.log = logger.Logger(verbose=self.verbose)
        else:
            self.fh = open(f"{filename}.txt", "w")
            self.log = logger.Logger(self.fh, verbose=self.verbose)

        if chkfile is not None:
            self.chkfile = chkfile
        
            if os.path.exists(self.chkfile):
                restart_from_chkfile(self)
            else:
                self.current_time = 0
        else:
            if filename is None:
                self.chkfile = "RT_CHKFILE.txt"
            else:
                self.chkfile = f"{filename}_CHKFILE.txt"
            
            self.current_time = 0
        
        self.den_ao = self._scf.make_rdm1(mo_occ=self.occ)
        rt_observables.init_observables(self)

    def istype(self, type_code):
        if isinstance(type_code, type):
            return isinstance(self, type_code)

        return any(type_code == t.__name__ for t in self.__class__.__mro__)

    def get_fock_orth(self, den_ao):
        self.fock = self._scf.get_fock(dm=den_ao).astype(np.complex128)
        if self.potential: self.applypotential()
        return np.matmul(self.orth.T, np.matmul(self.fock, self.orth))

    def rotate_coeff_to_orth(self, coeff_ao):
        return np.matmul(inv(self.orth), coeff_ao)

    def rotate_coeff_to_ao(self, coeff_orth):
        return np.matmul(self.orth, coeff_orth)

    def add_potential(self, *args):
        for v_ext in args:
            self.potential.append(v_ext)

    def applypotential(self):
        for v_ext in self.potential:
            self.fock += v_ext.calculate_potential(self)

    def kernel(self, mo_coeff_print=None):
        try:
            self.log.note("Starting Propagation")
            rt_scf_prop.propagate(self, mo_coeff_print)
        except Exception:
            raise
        finally:
            self.log.note("Done")
            if hasattr(self, 'fh'):
                self.fh.close()

        return self
