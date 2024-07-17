import numpy as np
from pyscf import gto, dft, scf, grad
import rt_integrators
import rt_observables
import rt_output
import rt_cap
import rt_vapp
import rt_nuclei
import ehrenfest_force
import rt_cpa
#from ehrenfest_brute_force import EhrenfestBruteForce
#from basis_utils import translatebasis
from rt_scf import RT_SCF

'''
Real-time SCF + Ehrenfest
'''

class RT_EHRENFEST(RT_SCF):
    def __init__(self, mf, timestep, frequency, total_steps, filename, prop=None, orth=None, Ne_step=10, N_step=10):

        super().__init__(mf, timestep, frequency, total_steps, filename, prop, orth)
        self.Ne_step = Ne_step
        self.N_step = N_step

        self.nuc = rt_nuclei.rt_nuc(self._scf.mol)
        self.den_ao = self._scf.make_rdm1(mo_occ = self.occ)
        if self.den_ao.dtype != np.complex128:
            self.den_ao = self.den_ao.astype(np.complex128)
        self.nuc.force = ehrenfest_force.get_force_new(self._scf, self.den_ao)
 
        self.t = 0
        #rt_observables.init_observables(self)

#    def set_grad(self):
#        if self._scf.istype('RKS'):
#            self._grad = self._scf.apply(grad.RKS)
#        elif self._scf.istype('RHF'):
#            self._grad = self._scf.apply(grad.RHF)
#        elif self._scf.istype('UKS'):
#            self._grad = self._scf.apply(grad.UKS)
#        elif self._scf.istype('UHF'):
#            self._grad = self._scf.apply(grad.UHF)
#        elif self._scf.istype('GKS'):
#            self._grad = self._scf.apply(grad.GKS)
#        elif self._scf.istype('GHF'):
#            self._grad = self._scf.apply(grad.GHF)
#        else:
#            raise Exception('unknown scf method')

#    def get_force(self):
#        #self.set_grad()
#        #self.nuc.force = ehrenfest_force.get_force(self._grad)
#        scf_copy = self._scf
#        self.nuc.force = ehrenfest_force.get_force_new(scf_copy)

    def get_fock_orth(self, den_ao=None):
        F = np.asarray(scf.hf.get_hcore(self._scf.mol)) + scf.uhf.get_veff(self._scf.mol, self.den_ao)
        iO = rt_cpa.get_fock_pert(self)
        F[0] += iO; F[1] += iO
        return np.array([np.linalg.multi_dot([self.orth.T, F[0], self.orth]), np.linalg.multi_dot([self.orth.T, F[1], self.orth])])
        #F = self._scf.get_fock(dm=self.den_ao)
        #retun np.matmul(self.orth.T, np.matmul(F, self.orth))

    def kernel_new(self, abinit=False):
        self.temp_create_output_file()

        self._scf = scf.UHF(self.nuc.get_mol())
        self._scf.mo_occ = self.occ
        #holds C(t-dt) between calculations:
        den_ao_old = self.den_ao
        for i in range(0, self.total_steps):
            #first velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)
            for j in range(0, self.N_step):
                #first position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
                #from current position update mol, overlap, and orthogonalization matrices:
                self._scf.mol = self.nuc.get_mol(); self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
                for k in range(0, self.Ne_step):
                    #P(t+dt) = U(t)P(t-dt)U*(t); U(t) = exp(i*2dt*F)                  
                    den_ao_old = rt_integrators.magnus_step_new(self, den_ao_old)
                    self.t += self.timestep
                    #print to output file check:
                    if np.mod(i * self.N_step * self.Ne_step + j * self.Ne_step + k, self.frequency) == 0:
                        self.temp_update_output_file()
                        norm = (np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao[0], np.linalg.inv(self.orth.T)]))
                                + np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao[1], np.linalg.inv(self.orth.T)])))
                        print(norm)
                        #print(f'Cmat: {self._scf.mo_coeff}')
                #second position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
            #before updating forces update mol object to current posiion;
            #self._scf.mol = self.nuc.get_mol(); self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
            ##### ABINITO START #####
            if abinit:
                mol_ab = self.nuc.get_mol()
                scf_ab = scf.UHF(mol_ab)
                scf_ab.mo_occ = self.occ
                scf_ab.kernel()
                self.nuc.force = ehrenfest_force.get_force_new(scf_ab)
            ##### ABINITO END #####
            #update force:
            else:
                self.nuc.force = ehrenfest_force.get_force_new(self._scf, self.den_ao)
            #second velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)

        return self

    def kernel_test(self):
        t = 0.
        x = 0.
        v = 0.
        f = 0.
        p = (0., 0., 0.)
        dt = 1. / (self.Ne_step * self.N_step)
        for i in range(0, 3):
            #first velocity half step:
            v += 0.5 * dt * self.Ne_step * self.N_step
            for j in range(0, self.N_step):
                #first position half step:
                x += 0.5 * dt * self.Ne_step
                #from current position update mol, overlap, and orthogonalization matrices:
                for k in range(0, self.Ne_step):
                    t += dt
                    p = (t, x, v)
                    print(f'p = {p}')
                #second position half step:
                x += 0.5 * dt * self.Ne_step
            #before updating forces update mol object to current posiion;
            p = (t, x, v)
            #update force:
            f = p
            print(f'f = {f}')
            #second velocity half step:
            v += 0.5 * dt * self.Ne_step * self.N_step
            print('----------------------------------------------------------')

        return self

    def kernel(self):
        self.temp_create_output_file()

        #holds C(t-dt) between calculations:
        mo_coeff_old = self._scf.mo_coeff
        for i in range(0, self.total_steps):
            #first velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)
            for j in range(0, self.N_step):
                #first position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
                #from current position update mol, overlap, and orthogonalization matrices:
                self._scf.mol = self.nuc.get_mol(); self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
                ##### JUNK TEST #####
                #self._scf.mol = self.nuc.get_mol(); self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
                #mo_coeff_copy = self._scf.mo_coeff
                #mol_copy = self.nuc.get_mol()
                #self.ovlp = mol_copy.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
                #self._scf = scf.UHF(mol_copy)
                #self._scf.mo_occ = self.occ
                #self._scf.kernel()
                #self._scf.mo_occ = self.occ
                #self._scf.mo_coeff = mo_coeff_copy
                ##### JUNK TEST #####
                for k in range(0, self.Ne_step):
                    #C(t+dt) = U(t)C(t-dt); U(t) = exp(-i*2dt*F)                  
                    mo_coeff_old = rt_integrators.magnus_step(self, mo_coeff_old)
                    self.t += self.timestep
                    norm = (np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao[0], np.linalg.inv(self.orth.T)]))
                            + np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao[1], np.linalg.inv(self.orth.T)])))
                    #print to output file check:
                    if np.mod(i * self.N_step * self.Ne_step + j * self.Ne_step + k, self.frequency) == 0:
                        self.temp_update_output_file()
                        print(norm)
                #second position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
            #before updating forces update mol object to current posiion;
            self._scf.mol = self.nuc.get_mol(); self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
            ##### JUNK TEST #####
            #self._scf.mol = self.nuc.get_mol(); self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
            #mo_coeff_copy = self._scf.mo_coeff
            #mol_copy = self.nuc.get_mol()
            #self.ovlp = mol_copy.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
            #self._scf = scf.UHF(mol_copy)
            #self._scf.mo_occ = self.occ
            #self._scf.kernel()
            #self._scf.mo_occ = self.occ
            #self._scf.mo_coeff = mo_coeff_copy
            ##### JUNK TEST #####
            #update force:
            self.get_force()
            ##### JUNK TEST #####
            #scf_copy = self._scf
            #scf_copy.mo_coeff = self._scf.mo_coeff.real + 1j * self._scf.mo_coeff.imag
            #scf_copy.mo_coeff = self._scf.mo_coeff.real
            #self.force_test = ehrenfest_force.get_force_new(scf_copy)
            ##### JUNK TEST #####
            #second velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)

        return self

    def kernel_abinit(self):
        self.temp_create_output_file()

        for i in range(0, self.total_steps):
            #first velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)
            for j in range(0, self.N_step):
                #first position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
                #from current position update mol, overlap, and orthogonalization matrices:
                self._scf = scf.UHF(self.nuc.get_mol()); self.ovlp = self._scf.mol.intor("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
                self._scf.mo_occ = self.occ
                self._scf.kernel()
                #self.den_ao = self._scf.make_rdm1(mo_occ = self.occ)
                for k in range(0, self.Ne_step):
                    self.t += self.timestep
                    #print to output file check:
                    if np.mod(i * self.N_step * self.Ne_step + j * self.Ne_step + k, self.frequency) == 0:
                        self.temp_update_output_file()
                        norm = (np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao[0], np.linalg.inv(self.orth.T)]))
                                + np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao[1], np.linalg.inv(self.orth.T)])))
                        print(norm)
                #second position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
            #before updating forces update mol object to current posiion;
            self._scf = scf.UHF(self.nuc.get_mol()); self.ovlp = self._scf.mol.intor("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
            self._scf.mo_occ = self.occ
            self._scf.kernel()
            #self.den_ao = self._scf.make_rdm1(mo_occ = self.occ)
            #update force:
            self.get_force()
            #second velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)

        return self

    def kernel_cpa(self):
        self.temp_create_output_file()

        #holds C(t-dt) between calculations:
        mo_coeff_old = self._scf.mo_coeff
        for i in range(0, self.total_steps):
            #first velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)
            for j in range(0, self.N_step):
                #first position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
                #from current position update mol, overlap, and orthogonalization matrices:
                self._scf.mol = self.nuc.get_mol(); self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
                for k in range(0, self.Ne_step):
                    #C(t+dt) = U(t)C(t-dt); U(t) = exp(-i*2dt*F)                  
                    mo_coeff_old = rt_integrators.magnus_step(self, mo_coeff_old)
                    self.t += self.timestep
                    #print to output file check:
                    if np.mod(i * self.N_step * self.Ne_step + j * self.Ne_step + k, self.frequency) == 0:
                        self.temp_update_output_file()
                        norm = (np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao[0], np.linalg.inv(self.orth.T)]))
                                + np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao[1], np.linalg.inv(self.orth.T)])))
                        print(norm)
                #second position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
            #before updating forces update mol object to current posiion;
            self._scf.mol = self.nuc.get_mol(); self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
            ##### ABINITO START #####
            mol_ab = self.nuc.get_mol()
            scf_ab = scf.UHF(mol_ab)
            scf_ab.mo_occ = self.occ
            scf_ab.kernel()
            ##### ABINITO END #####
            #update force:
            self.nuc.force = ehrenfest_force.get_force_new(scf_ab)
            #second velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)

        return self

    def kernel2(self):
        self.temp_create_output_file()

        #holds C(t-dt) between calculations:
        #mo_coeff_old = self._scf.mo_coeff
        for i in range(0, self.total_steps):
            #first velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)
            for j in range(0, self.N_step):
                #first position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
                #from current position update mol, overlap, and orthogonalization matrices:
                #self._scf = scf.UHF(self.nuc.get_mol())
                #self._scf.kernel()
                #mo_coeff_old = self._scf.mo_coeff
                #self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
                for k in range(0, self.Ne_step):
                    #C(t+dt) = U(t)C(t-dt); U(t) = exp(-i*2dt*F)                  
                    #mo_coeff_old = rt_integrators.magnus_step(self, mo_coeff_old)
                    self.t += self.timestep
                    #print to output file check:
                    if np.mod(i * self.Ne_step * self.N_step + j * self.N_step + k + 1, self.frequency) == 0:
                        self.temp_update_output_file()
                #second position half step:
                self.nuc.get_pos(self.timestep * self.Ne_step)
            #before updating forces update mol object to current posiion;
            self._scf = scf.UHF(self.nuc.get_mol())
            self._scf.kernel()
            mo_coeff_old = self._scf.mo_coeff
            self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
            for l in range(0, self.Ne_step * self.N_step):
                mo_coeff_old = rt_integrators.magnus_step(self, mo_coeff_old)
            #update force:
            self.get_force()
            #second velocity half step:
            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)

        return self

#    def kernel2(self):
#        self.temp_create_output_file()
#
#        #holds C(t-dt) between calculations:
#        mo_coeff_old = self._scf.mo_coeff
#        for i in range(0, self.total_steps):
#            #first velocity half step:
#            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)
#            for j in range(0, self.N_step):
#                #first position half step:
#                self.nuc.get_pos(self.timestep * self.Ne_step)
#                #from current position update mol, overlap, and orthogonalization matrices:
#                mo_coeff_new = self._scf.mo_coeff
#                self._scf = scf.UHF(self.nuc.get_mol())
#                self._scf.kernel()
#                self._scf.mo_coeff = mo_coeff_new
#                self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
#                for k in range(0, self.Ne_step):
#                    #C(t+dt) = U(t)C(t-dt); U(t) = exp(-i*2dt*F)                  
#                    mo_coeff_old = rt_integrators.magnus_step(self, mo_coeff_old)
#                    self.t += self.timestep
#                    #print to output file check:
#                    if np.mod(i * self.Ne_step * self.N_step + j * self.N_step + k + 1, self.frequency) == 0:
#                        self.temp_update_output_file()
#                #second position half step:
#                self.nuc.get_pos(self.timestep * self.Ne_step)
#            #before updating forces update mol object to current posiion;
#            mo_coeff_new = self._scf.mo_coeff
#            self._scf = scf.UHF(self.nuc.get_mol())
#            self._scf.kernel()
#            self._scf.mo_coeff = mo_coeff_new
#            self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp"); self.orth = scf.addons.canonical_orth_(self.ovlp)
#            #update force:
#            self.get_force()
#            #second velocity half step:
#            self.nuc.get_vel(self.timestep * self.Ne_step * self.N_step)
#
#        return self

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
        np.savetxt(pos_file, self.nuc.pos, '%20.8e')
        pos_file.write('\n')
        np.savetxt(vel_file, self.nuc.vel, '%20.8e')
        vel_file.write('\n')
        np.savetxt(force_file, self.nuc.force, '%20.8e')
        force_file.write('\n')
        t_out = self.t
        Etot = self._scf.energy_tot(dm=self.den_ao) + self.nuc.get_ke()
        Eelec = self._scf.energy_elec(dm=self.den_ao)[0]
        Vnuc = self._scf.energy_nuc()
        Tnuc = self.nuc.get_ke()
        print(t_out)
        print(Etot)
        print(Eelec)
        print(Vnuc)
        print(Tnuc)
        output_ar = np.array([[t_out,Etot,Eelec,Vnuc,Tnuc]])
        np.savetxt(energy_file, output_ar, '%20.8e')
        energy_file.flush()
        #energy_file.write('-iOmega:')
        #iO = rt_cpa.get_fock_pert(self)
        #energy_file.write(f'{iO + np.conj(iO.T)}')
        #energy_file.write('\n')
        #energy_file.write('F:')
        #F = self._scf.get_fock()
        #energy_file.write(f'{F[0] - np.conj(F[0].T)}')
        #energy_file.write('\n')
        #energy_file.write('den:')
        #energy_file.write(f'{self.den_ao}')
        #energy_file.write('\n')
        pos_file.close()
        vel_file.close()
        force_file.close()
        energy_file.close()

            #if np.mod(i, self.frequency) == 0:
                #rt_observables.get_observables(self.rt_scf, mo_coeff_print)
                #self.temp_update_output_file()
                #self.h1e = scf.hf.get_hcore(self._scf.mol)
                #self.vhf = scf.hf.get_veff(self._scf.mol, self.den_ao)
                #print(scf.hf.energy_tot(self._scf, dm=self.den_ao, h1e=self.h1e, vhf=self.vhf) + self.nuc.get_ke())
                #print(self._scf.energy_tot() + self.nuc.get_ke())
                #print('---------------------------------------------\n')
                #print(self.den_ao)
                #print('---------------------------------------------\n')


