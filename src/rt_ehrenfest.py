import numpy as np
from pyscf import gto, dft, scf, grad
import rt_integrators
import rt_observables
import rt_output
from rt_utils import update_chkfile, update_fragments
#from rt_utils import restart_from_chkfile
#from pathlib import Path
import ehrenfest_force
from rt_scf import RT_SCF
from rt_nuclei import NUC

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
        self.nuc = NUC(self._scf.mol)

        self.current_time = 0

        #reminder to use fractional_matrix_power and not the rounded off pyscf function
        self.update_mol()
        self.update_grad()
        self.nuc.force = ehrenfest_force.get_force(self)

    #for some reason updating the mol object does not update the 2 electron integrals so we're temporarily reinstantiating
    def update_mol(self):
        mo_coeff = self._scf.mo_coeff
        if self._scf.istype('RHF'): self._scf = scf.RHF(self.nuc.get_mol())
        elif self._scf.istype('RKS'): self._scf = scf.RKS(self.nuc.get_mol())
        elif self._scf.istype('UHF'): self._scf = scf.UHF(self.nuc.get_mol())
        elif self._scf.istype('UKS'): self._scf = scf.UKS(self.nuc.get_mol())
        elif self._scf.istype('GHF'): self._scf = scf.GHF(self.nuc.get_mol())
        elif self._scf.istype('GKS'): self._scf = scf.GKS(self.nuc.get_mol())
        self._scf.mo_coeff = mo_coeff
        self._scf.mo_occ = self.occ
        #we should add a routine to RT_SCF maybe for setting the overlap, orthogonalization matrix and maybe holding onto eigen stuff from S matrix
        self.ovlp = self._scf.mol.intor_symmetric("int1e_ovlp")
        self.evals, self.evecs = np.linalg.eigh(self.ovlp)
        self.orth = np.linalg.multi_dot([self.evecs, np.diag(np.power(self.evals, -0.5)), self.evecs.T])
        #self.orth = fractional_matrix_power(self.ovlp, -0.5)
        #self.orth = scf.addons.canonical_orth_(self.ovlp)

    def update_grad(self):
        if self._scf.istype('RHF'): self._grad = self._scf.apply(grad.RHF)
        elif self._scf.istype('RKS'): self._grad = self._scf.apply(grad.RKS)
        elif self._scf.istype('UHF'): self._grad = self._scf.apply(grad.UHF)
        elif self._scf.istype('UKS'): self._grad = self._scf.apply(grad.UKS)
        elif self._scf.istype('GHF'): self._grad = self._scf.apply(grad.GHF)
        elif self._scf.istype('GKS'): self._grad = self._scf.apply(grad.GKS)

    # Additinal term arising from the moving nuclei in the classical path approximation to be added to the fock matrix
    def get_omega(self):
        mol = self._scf.mol
        Rdot = self.nuc.vel
        X = self.orth
        Xinv = np.linalg.inv(X)
        dS = -mol.intor('int1e_ipovlp', comp=3)
    
        Omega = np.zeros(X.shape, dtype = complex)
        RdSX = np.zeros(X.shape)
        aoslices = mol.aoslice_by_atom()
        for i in range(mol.natm):
            p0, p1 = aoslices[i,2:]
            RdSX += np.einsum('x,xij,ik->jk', Rdot[i], dS[:,p0:p1,:], X[p0:p1,:])
        Omega += np.matmul(RdSX, Xinv)
        return Omega

    def get_fock_orth(self, den_ao):
        Omega = self.get_omega()
        self.fock = self._scf.get_fock(dm=den_ao)
        if self.potential: self.applypotential()
        #return np.matmul(self.orth.T, np.matmul(self.fock - 1j * Omega, self.orth))
        return np.matmul(self.orth.T, np.matmul(self.fock, self.orth))

    def rotate_coeff_to_orth(self, coeff_ao):
        return np.matmul(np.linalg.inv(self.orth), coeff_ao)

    def rotate_coeff_to_ao(self, coeff_orth):
        return np.matmul(self.orth, coeff_orth)

    def kernel(self, mo_coeff_print=None, match_indices_array=None):
        rt_observables.remove_suppressed_observables(self)
        self.temp_create_output_file()
 
        integrate_function = rt_integrators.get_integrator(self)
        if self.prop == "magnus_step":
            self.fock_orth = self.get_fock_orth(self.den_ao)
            self.mo_coeff_orth_old = self.rotate_coeff_to_orth(self._scf.mo_coeff)
        if self.prop == "magnus_interpol_temp":
            self.fock_orth = self.get_fock_orth(self.den_ao)
            self.fock_orth_n12dt = self.get_fock_orth(self.den_ao)
            self.mo_coeff_orth = self.rotate_coeff_to_orth(self._scf.mo_coeff)
            if not hasattr(self, 'magnus_tolerance'): self.magnus_tolerance = 1e-7
            if not hasattr(self, 'magnus_maxiter'): self.magnus_maxiter = 15
        if self.prop == "rk4":
            self.fock_orth = self.get_fock_orth(self.den_ao)
            self.mo_coeff_orth = self.rotate_coeff_to_orth(self._scf.mo_coeff)
    
        for i in range(0, self.total_steps):
            if np.mod(i, self.frequency) == 0:
                self.temp_update_output_file()
                mo_coeff_print = update_fragments(self, match_indices_array) 
                rt_observables.get_observables(self, mo_coeff_print)
                update_chkfile(self)
            self.nuc.get_vel(0.5 * self.Ne_step * self.N_step * self.timestep)
            for j in range(0, self.N_step):
                self.nuc.get_pos(0.5 * self.Ne_step * self.timestep)
                self.update_mol()
                for k in range(0, self.Ne_step):
                    if (k == self.Ne_step - 1):
                        self.nuc.get_pos(0.5 * self.Ne_step * self.timestep)
                        self.update_mol()                   
                        if (j == self.N_step - 1):
                            self.nuc.get_vel(0.5 * self.Ne_step * self.N_step * self.timestep)
                    integrate_function(self)
                    self.current_time += self.timestep
            self.update_grad()
            self.nuc.get_vel(-0.5 * self.Ne_step * self.N_step * self.timestep)
            self.nuc.force = ehrenfest_force.get_force(self)
            self.nuc.get_vel(0.5 * self.Ne_step * self.N_step * self.timestep)
        self.temp_update_output_file()
        mo_coeff_print = update_fragments(self, match_indices_array) 
        rt_observables.get_observables(self, mo_coeff_print)  # Collect observables at final time
        update_chkfile(self)
        return self

# kernel_simple() should simplify to kernel() if Ne_step = N_step = 1
    def kernel_simple(self):
        rt_observables.remove_suppressed_observables(self)
        self.temp_create_output_file()

        integrate_function = rt_integrators.get_integrator(self)
        if self.prop == "magnus_step":
            self.fock_orth = self.get_fock_orth(self.den_ao)
            self.mo_coeff_orth_old = self.rotate_coeff_to_orth(self._scf.mo_coeff)
        if self.prop == "magnus_interpol_temp":
            self.fock_orth = self.get_fock_orth(self.den_ao)
            self.fock_orth_n12dt = self.get_fock_orth(self.den_ao)
            self.mo_coeff_orth = self.rotate_coeff_to_orth(self._scf.mo_coeff)
            if not hasattr(self, 'magnus_tolerance'): self.magnus_tolerance = 1e-7
            if not hasattr(self, 'magnus_maxiter'): self.magnus_maxiter = 15
        if self.prop == "rk4":
            self.fock_orth = self.get_fock_orth(self.den_ao)
            self.mo_coeff_orth = self.rotate_coeff_to_orth(self._scf.mo_coeff)
    
        for i in range(0, self.total_steps):
            if np.mod(i, self.frequency) == 0:
                self.temp_update_output_file()
                #rt_observables.get_observables(self, mo_coeff_print)
                update_chkfile(self)

            self.nuc.get_vel(0.5 * self.timestep)
            self.nuc.get_pos(self.timestep)
            self.update_mol()
            self.nuc.get_vel(0.5 * self.timestep)
            integrate_function(self)
            self.nuc.get_vel(-0.5 * self.timestep)
            self.nuc.force = ehrenfest_force.get_force(self)
            self.nuc.get_vel(0.5 * self.timestep)

            self.current_time += self.timestep
    
        self.temp_update_output_file()
        #rt_observables.get_observables(self, mo_coeff_print)  # Collect observables at final time
        update_chkfile(self)
        return self


# EVERYTHING BELOW IS JUNK!!!
#
#    def kernel_slice1(self):
#        rt_observables.remove_suppressed_observables(self)
#        self.temp_create_output_file()
#    
#        integrate_function = rt_integrators.get_integrator(self)
#        if self.prop == "magnus_step":
#            self.fock_orth = self.get_fock_orth(self.den_ao)
#            self.mo_coeff_orth_old = self.rotate_coeff_to_orth(self._scf.mo_coeff)
#        if self.prop == "magnus_interpol_temp":
#            self.fock_orth = self.get_fock_orth(self.den_ao)
#            self.fock_orth_n12dt = self.get_fock_orth(self.den_ao)
#            self.mo_coeff_orth = self.rotate_coeff_to_orth(self._scf.mo_coeff)
#            if not hasattr(self, 'magnus_tolerance'): self.magnus_tolerance = 1e-7
#            if not hasattr(self, 'magnus_maxiter'): self.magnus_maxiter = 15
#        if self.prop == "rk4":
#            self.fock_orth = self.get_fock_orth(self.den_ao)
#            self.mo_coeff_orth = self.rotate_coeff_to_orth(self._scf.mo_coeff)
#    
#        for i in range(0, self.total_steps):
#            if np.mod(i, self.frequency) == 0:
#                self.temp_update_output_file()
#                #rt_observables.get_observables(self, mo_coeff_print)
#                update_chkfile(self)
#
#            self.nuc.get_vel(0.5 * self.Ne_step * self.timestep)
#            self.nuc.get_pos(0.5 * self.Ne_step * self.timestep)
#            self.update_mol()
#            for k in range(0, self.Ne_step):
#                if (k == self.Ne_step - 1):
#                    self.nuc.get_pos(0.5 * self.Ne_step * self.timestep)
#                    self.update_mol()                   
#                    self.nuc.get_vel(0.5 * self.Ne_step * self.timestep)
#                integrate_function(self)
#                self.current_time += self.timestep
#            self.nuc.get_vel(-0.5 * self.Ne_step * self.timestep)
#            self.nuc.force = ehrenfest_force.get_force(self)
#            self.nuc.get_vel(0.5 * self.Ne_step * self.timestep)
#
#        self.temp_update_output_file()
#        #rt_observables.get_observables(self, mo_coeff_print)  # Collect observables at final time
#        update_chkfile(self)
#        return self
#
#    def kernel_interpol(self):
#        self.temp_create_output_file()
#        self.mo_coeff_orth = np.matmul(np.linalg.inv(self.orth), self._scf.mo_coeff)
#        self.fock_orth = self.get_fock_orth(self.den_ao)
#        self.fock_orth_n12dt = self.fock_orth
#        if not hasattr(self, 'magnus_tolerance'): self.magnus_tolerance = 1e-7
#        if not hasattr(self, 'magnus_maxiter'): self.magnus_maxiter = 15
#        for i in range(0, self.total_steps):
#            #print to output file check:
#            if np.mod(i, self.frequency) == 0:
#                self.temp_update_output_file()
#            #fock get fock at current time step and nuclear geometry
#            self.fock_orth = self.get_fock_orth(self.den_ao)
#            #first velocity half step:
#            self.nuc.get_vel(self.timestep)
#            #position full step:
#            self.nuc.get_pos(self.timestep)
#            self.nuc.get_pos(self.timestep)           
#            #from current position update mol, overlap, and orthogonalization matrices:
#            self.update_mol()
#            self.orth_old = self.orth
#
#            fock_orth_p12dt = 2 * self.fock_orth - self.fock_orth_n12dt
#            for iteration in range(self.magnus_maxiter):
#                u = expm(-1j*self.timestep*fock_orth_p12dt)
# 
#                mo_coeff_orth_pdt = np.matmul(u, self.mo_coeff_orth)
#                mo_coeff_ao_pdt = np.matmul(self.orth, mo_coeff_orth_pdt)
#                den_ao_pdt = self._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt, mo_occ=self.occ)
#
#                if (iteration > 0 and
#                abs(np.linalg.norm(mo_coeff_ao_pdt)
#                - np.linalg.norm(mo_coeff_ao_pdt_old)) < self.magnus_tolerance):
#        
#                    self.mo_coeff_orth = mo_coeff_orth_pdt
#                    self._scf.mo_coeff = mo_coeff_ao_pdt
#                    self.den_ao = den_ao_pdt
#        
#                    self.fock_orth_n12dt = fock_orth_p12dt
#                    break
#        
#                fock_orth_pdt = self.get_fock_orth(den_ao_pdt)
#        
#                fock_orth_p12dt = 0.5 * (self.fock_orth + fock_orth_pdt)
#        
#                mo_coeff_ao_pdt_old = mo_coeff_ao_pdt
#        
#                self._scf.mo_coeff = mo_coeff_ao_pdt
#                self.den_ao = den_ao_pdt
#            #update force:
#            self.nuc.force = ehrenfest_force.get_force(self)
#            #second velocity half step:
#            self.nuc.get_vel(self.timestep)
#            self.current_time += self.timestep
#        return self
#
#    def kernel_step(self):
#        self.temp_create_output_file()
#        fock_orth = self.get_fock_orth(self.den_ao)
#        fock_orth_old = fock_orth
#        udag = expm(1j * self.timestep * fock_orth)
#        #holding C(t-dt) and C(t+dt) between calculations:
#        mo_coeff_orth_new = np.matmul(np.linalg.inv(self.orth), self._scf.mo_coeff)
#        mo_coeff_orth_old = np.matmul(udag, mo_coeff_orth_new)
#        for i in range(0, self.total_steps):
#            #print to output file check:
#            if np.mod(i, self.frequency) == 0:
#                self.temp_update_output_file()
#                fock_orth_old = fock_orth
#                fock_orth = self.get_fock_orth(self.den_ao)
#                print('drive:')
#                drive = self.get_drive(mo_coeff_orth_old, fock_orth_old, fock_orth)
#                print(drive.shape)
#                print(drive)
#                #print(fock_orth)
#                print('den:')
#                print(self.den_ao)
#            #first velocity half step:
#            self.nuc.get_vel(self.timestep)
#            #fock get fock at current time step and nuclear geometry
#            fock_orth = self.get_fock_orth(self.den_ao)
#            #propagator U(t) = exp(-i*2dt*F')
#            u = expm(-1j * 2 * self.timestep * fock_orth)
#            #integrate C'(t+dt) = U(t)C'(t-dt)
#            mo_coeff_orth_new = np.matmul(u, mo_coeff_orth_old)
#            #from current C(t) and current X(t), get C'(t-dt) for next loop
#            mo_coeff_orth_old = np.matmul(np.linalg.inv(self.orth), self._scf.mo_coeff)
#            #position full step:
#            self.nuc.get_pos(self.timestep)
#            self.nuc.get_pos(self.timestep)
#            #from current position update mol, overlap, and orthogonalization matrices:
#            self.update_mol()
#            #from new C'(t+dt) and new X(t+dt), get C(t+dt)/P(t+dt)
#            self._scf.mo_coeff = np.matmul(self.orth, mo_coeff_orth_new)
#            self.den_ao = self._scf.make_rdm1(mo_coeff = self._scf.mo_coeff, mo_occ = self._scf.mo_occ)
#            #update force:
#            self.nuc.force = ehrenfest_force.get_force(self)
#            #second velocity half step:
#            self.nuc.get_vel(self.timestep)
#            self.current_time += self.timestep
#        return self
#
#    def kernel_drive(self):
#        self.temp_create_output_file()
#        fock_orth = self.get_fock_orth(self.den_ao)
#        fock_orth_new = fock_orth 
#        mo_coeff_orth = np.matmul(np.linalg.inv(self.orth), self._scf.mo_coeff)
#        mo_coeff_orth_new = mo_coeff_orth 
#        for i in range(0, self.total_steps):
#            #print to output file check:
#            if np.mod(i, self.frequency) == 0:
#                self.temp_update_output_file()
#            #first velocity half step:
#            self.nuc.get_vel(self.timestep)
#            #position full step:
#            self.nuc.get_pos(self.timestep)
#            self.nuc.get_pos(self.timestep)
#            self.update_mol()
#            u = expm(-1j * self.timestep * fock_orth)
#            for n in range(5): 
#                if n==0:
#                    fock_orth_new = fock_orth
#                else:
#                    fock_orth_new = self.get_fock_orth(self.den_ao)
#                drive_orth = self.get_drive(mo_coeff_orth, fock_orth, fock_orth_new)
#                mo_coeff_orth_new = np.matmul(u, mo_coeff_orth) - np.matmul(u, np.matmul(drive_orth, mo_coeff_orth))
#                self._scf.mo_coeff = np.matmul(self.orth, mo_coeff_orth_new)
#                self.den_ao = self._scf.make_rdm1(mo_coeff = self._scf.mo_coeff, mo_occ = self._scf.mo_occ)
#            fock_orth = self.get_fock_orth(self.den_ao)
#            mo_coeff_orth = mo_coeff_orth_new
#            #update force:
#            self.nuc.force = ehrenfest_force.get_force(self)
#            #second velocity half step:
#            self.nuc.get_vel(self.timestep)
#            self.current_time += self.timestep
#        return self
#
#    def kernel_abinit(self):
#        self.temp_create_output_file()
#        for i in range(0, self.total_steps):
#            #print to output file check:
#            if np.mod(i, self.frequency) == 0:
#                self.temp_update_output_file()
#            #first velocity half step:
#            self.nuc.get_vel(self.timestep)
#            #position full step:
#            self.nuc.get_pos(self.timestep)
#            self.nuc.get_pos(self.timestep)
#            #from current position update mol, overlap, and orthogonalization matrices:
#            self.update_mol()
#            self._scf.kernel() 
#            self.den_ao = self._scf.make_rdm1(mo_coeff = self._scf.mo_coeff, mo_occ = self._scf.mo_occ)
#            #update force:
#            self.nuc.force = ehrenfest_force.get_force(self)
#            #second velocity half step:
#            self.nuc.get_vel(self.timestep)
#            self.current_time += self.timestep
#        return self
#
##    def get_drive(self, mo_coeff_orth, fock_orth, fock_orth_new, pts=69):
##        tau_ar = np.linspace(0., self.timestep, pts)
##        Nbasis = mo_coeff_orth.shape[0]
##        drive_ar = np.zeros((pts, Nbasis, Nbasis), dtype=complex)
##        for i in range(pts):
##            drive_ar[i] = np.linalg.multi_dot([expm(1j * tau_ar[i] * fock_orth), fock_orth_new - fock_orth, expm(-1j * tau_ar[i] * fock_orth)]) * tau_ar[i]
##        return -1j / self.timestep * np.linalg.multi_dot([expm(-1j * self.timestep * fock_orth), np.trapz(drive_ar, tau_ar, axis=0), mo_coeff_orth])
#
#    def get_drive(self, mo_coeff_orth, fock_orth, fock_orth_new, pts=9969):
#        tau_ar = np.linspace(0., self.timestep, pts)
#        Nbasis = mo_coeff_orth.shape[-1]
#        drive_ar = np.zeros((pts, 2, Nbasis, Nbasis), dtype=complex)
#        print(drive_ar.shape)
#        print(drive_ar[24,0].shape)
#        for i in range(pts):
#            drive_ar[i,0] = np.linalg.multi_dot([expm(1j * tau_ar[i] * fock_orth[0]), fock_orth_new[0] - fock_orth[0], expm(-1j * tau_ar[i] * fock_orth[0])]) * tau_ar[i]
#            drive_ar[i,1] = np.linalg.multi_dot([expm(1j * tau_ar[i] * fock_orth[1]), fock_orth_new[1] - fock_orth[1], expm(-1j * tau_ar[i] * fock_orth[1])]) * tau_ar[i]
#        return -1j / self.timestep * np.matmul(expm(-1j * self.timestep * fock_orth), np.matmul(np.trapz(drive_ar, tau_ar, axis=0), mo_coeff_orth))


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
        #norm = np.trace(np.linalg.multi_dot([np.linalg.inv(self.orth), self.den_ao, np.linalg.inv(self.orth.T)])); print(norm)
        t_out = self.current_time
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

