import numpy as np
from scipy.linalg import eigh, inv
from pyscf import gto, scf
import scipy
import matplotlib.pyplot as plt

# class needs mf, timestep, frequency, total_steps

class GHF:
    def __init__(self, mf, timestep, frequency, total_steps, filename, orth=None):
        self.timestep = timestep
        self.frequency = frequency 
        self.total_steps = total_steps
        self.filename = filename
        self._scf = mf
    
        if orth is None: self.orth = scf.addons.canonical_orth_(self._scf.get_ovlp())
 
    ####### DYNAMICS #######
    def dynamics(self):
        ### creating output file for observables (edit to be main output file and to adjust what is calculated)
        observables = open(F'{self.filename}.txt', 'w')
        with open(F'{self.filename}.txt', 'a') as f:
            observables.write('{0: >20}'.format('Time'))
            observables.write('{0: >35}'.format('Mag x')) 
            observables.write('{0: >36}'.format('Mag y')) 
            observables.write('{0: >37}'.format('Mag z'))
            observables.write('{0: >37}'.format('Energy'))
            observables.write(F'\n')

        ### creating initial core hamiltonian
        fock = self._scf.get_fock()
        mag_x = []
        mag_y = []
        mag_z = []
        t_array = []
        energy = []

        ovlp = self._scf.get_ovlp()
        hcore = self._scf.get_hcore()
        Nsp = int(ovlp.shape[0]/2)

        for i in range(0, self.total_steps):
            ### transforming coefficients into an orthogonal matrix 
            mo_oth = np.dot(inv(self.orth), self._scf.mo_coeff)

            ### create transformation matrix U from Fock matrix at time t 
            fock_oth = np.dot(self.orth.T, np.dot(fock, self.orth))

            u = scipy.linalg.expm(-1j*2*self.timestep*fock_oth) 

            ### propagate MO coefficients 
            if i != 0:
                mo_oth_new = np.dot(u, mo_oth_old)
            else:
                mo_oth_new = np.dot(u, mo_oth)

            ### transform coefficients back into non-orthogonal basis and get density matrix
            self._scf.mo_coeff = np.dot(self.orth, mo_oth_new)
            den = self._scf.make_rdm1() 

            # calculate a new fock matrix
            fock = self._scf.get_fock(hcore)

            # calculate energy and other observables
            if np.mod(i, self.frequency)==0:
                ener_tot = self._scf.energy_tot()

                den = self._scf.make_rdm1()

                mag_x_value = 0
                mag_y_value = 0
                mag_z_value = 0

                for k in range(0, Nsp):
                    for j in range(0, Nsp):
                        ab_add = den[:Nsp, Nsp:][k,j] + den[Nsp:, :Nsp][k,j]
                        mag_x_value += ab_add * ovlp[k,j]
        
                        ab_sub = den[:Nsp, Nsp:][k,j] - den[Nsp:, :Nsp][k,j]
                        mag_y_value += ab_sub * ovlp[k,j]

                        aa_bb = den[:Nsp, :Nsp][k,j] - den[Nsp:, Nsp:][k,j]
                        mag_z_value += aa_bb * ovlp[k,j]

                t = (i * self.timestep) / 41341.374575751

                with open(F'{self.filename}.txt', 'a') as f:
                    observables.write(F'{t:20.8e} \t {mag_x_value:20.8e} \t {mag_y_value:20.8e} \t {mag_z_value:20.8e} \t {ener_tot:20.8e} \n')

            mo_oth_old = mo_oth


    ####### PLOTTING RESULTS #######
    def plot_mag(self):

        table = []
        openfile = F'{self.filename}.txt'

        with open(openfile, 'r') as f:
            next(f)
            for line in f:
                data = line.split('\t')
                data = [x.strip() for x in data]
                table.append(data)

        table = np.asarray(table)

        plt.figure(1)
        plt.plot(table[:,0], np.real(table[:,1]), 'r', label='mag_x')
        plt.plot(table[:,0], np.real(table[:,2]), 'b', label='mag_y')
        plt.plot(table[:,0], np.real(table[:,3]), 'g', label='mag_z')
        plt.xlabel('Time (ps)')
        plt.ylabel('Magnetization (au)')
        plt.legend()
        plt.savefig(F'{self.filename}_mag.png') 



    def plot_energy(self):

        table = []
        openfile = F'{self.filename}.txt'

        with open(openfile, 'r') as f:
            next(f)
            for line in f:
                data = line.split('\t')
                data = [x.strip() for x in data]
                table.append(data)

        table = np.asarray(table)

        plt.figure(2)
        plt.plot(table[:,0], np.real(table[:,4]), 'r')
        plt.xlabel('Time (ps)')
        plt.ylabel('Energy (Hartrees)')
        plt.savefig(F'{self.filename}_energy.png')              
