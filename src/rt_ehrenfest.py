import numpy as np
from pyscf import gto, dft, scf, grad
#import rt_scf_prop
#import rt_observables
#import rt_output
#import rt_scf_cleanup
#from rt_utils import restart_from_chkfile
#from pathlib import Path
import rt_nuclei
import ehrenfest_force
import rt_cpa
from rt_scf import RT_SCF

from scipy.linalg import expm, inv, fractional_matrix_power

'''
Real-time SCF + Ehrenfest
'''

class RT_EHRENFEST(RT_SCF):
    def __init__(self, mf, timestep, frequency, total_steps, filename=None, prop=None, orth=None, chkfile=None, verbose=3, Ne_step=10, N_step=10):

        super().__init__(mf, timestep, frequency, total_steps, filename, prop, orth, chkfile, verbose)
        self.Ne_step = Ne_step
        self.N_step = N_step
        self.filename = filename

        self.den_ao = self._scf.make_rdm1(mo_coeff = self._scf.mo_coeff, mo_occ = self._scf.mo_occ)
        if self.den_ao.dtype != np.complex128:
            self.den_ao = self.den_ao.astype(np.complex128)
        self.nuc = rt_nuclei.rt_nuc(self._scf.mol)
        self.nuc.force = ehrenfest_force.get_force(self._scf, self.den_ao)
        self.t = 0
        #rt_observables.init_observables(self)

        #reminder to use fractional_matrix_power and not the rounded off pyscf function
        self.update_mol()

    def update_mol(self):
        self._scf.mol = self.nuc.get_mol()
        self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp")
        self.orth = fractional_matrix_power(self.ovlp, -0.5)
        #self.orth = scf.addons.canonical_orth_(self.ovlp)

    def get_fock_orth(self, den_ao):
        self.fock = self._scf.get_fock(dm=den_ao)
        #if self.movebasis: Omega = rt_cpa.get_omega(self)
        return np.matmul(self.orth.T, np.matmul(self.fock, self.orth))

    def kernel(self):
        self.temp_create_output_file()
        #holding C(t-dt) and C(t+dt) between calculations:
        mo_coeff_orth_old = np.matmul(np.linalg.inv(self.orth), self._scf.mo_coeff)
        mo_coeff_orth_new = mo_coeff_orth_old 
        for i in range(0, self.total_steps):
            #print to output file check:
            if np.mod(i, self.frequency) == 0:
                self.temp_update_output_file()
            #first velocity half step:
            self.nuc.get_vel(self.timestep)
            #fock get fock at current time step and nuclear geometry
            fock_orth = self.get_fock_orth(self.den_ao)
            #propagator U(t) = exp(-i*2dt*F')
            u = expm(-1j * 2 * self.timestep * fock_orth)
            #integrate C'(t+dt) = U(t)C'(t-dt)
            mo_coeff_orth_new = np.matmul(u, mo_coeff_orth_old)
            #from current C(t) and current X(t), get C'(t-dt) for next loop
            mo_coeff_orth_old = np.matmul(np.linalg.inv(self.orth), self._scf.mo_coeff)
            #position full step:
            self.nuc.get_pos(self.timestep)
            self.nuc.get_pos(self.timestep)
            #from current position update mol, overlap, and orthogonalization matrices:
            self.update_mol()
            #from new C'(t+dt) and new X(t+dt), get C(t+dt)/P(t+dt)
            self._scf.mo_coeff = np.matmul(self.orth, mo_coeff_orth_new)
            self.den_ao = self._scf.make_rdm1(mo_coeff = self._scf.mo_coeff, mo_occ = self._scf.mo_occ)
            #update force:
            self.nuc.force = ehrenfest_force.get_force(self._scf, self.den_ao)
            #second velocity half step:
            self.nuc.get_vel(self.timestep)
            self.t += self.timestep
        return self

    def temp_create_output_file(self):
        pos_file = open(F'{self.filename}' + '_pos.txt','w')
        vel_file = open(F'{self.filename}' + '_vel.txt','w')
        force_file = open(F'{self.filename}' + '_force.txt','w')
        energy_file = open(F'{self.filename}' + '_energy.txt','w')

        pos_file.close()
        vel_file.close()
        force_file.close()
        energy_file.close()

    def temp_update_output_file(self):
        pos_file = open(F'{self.filename}' + '_pos.txt','a')
        vel_file = open(F'{self.filename}' + '_vel.txt','a')
        force_file = open(F'{self.filename}' + '_force.txt','a')
        energy_file = open(F'{self.filename}' + '_energy.txt','a')

        np.savetxt(pos_file, self.nuc.pos, '%20.8e'); pos_file.write('\n')
        np.savetxt(vel_file, self.nuc.vel, '%20.8e'); vel_file.write('\n')
        np.savetxt(force_file, self.nuc.force, '%20.8e'); force_file.write('\n')
        t_out = self.t
        Eelec = self._scf.energy_elec(dm=self.den_ao)[0]
        Vnuc = self._scf.energy_nuc()
        Tnuc = self.nuc.get_ke()
        Etot = Eelec + Vnuc + Tnuc
        output_ar = np.array([[t_out,Etot,Eelec,Vnuc,Tnuc]])
        np.savetxt(energy_file, output_ar, '%20.8e')
        energy_file.flush()

        pos_file.close()
        vel_file.close()
        force_file.close()
        energy_file.close()

