import numpy as np
from pyscf import gto, dft, scf, grad
import rt_integrators
import rt_observables
import rt_output
import rt_cap
import rt_vapp
import rt_nuclei
#from basis_utils import translatebasis

from pyscf.lo.orth import lowdin

from scipy.linalg import fractional_matrix_power


#print(dV_ana1)

class EhrenfestBruteForce:
    def __init__(self, rt_nuc, displacement=1e-5):
        self.rt_nuc = rt_nuc
        self.displacement = displacement

    def get_grad_ovlp(self):
        pos_copy = self.rt_nuc.pos
        disp_ar = np.zeros((self.rt_nuc.nnuc, 3))

        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        dS_ar = np.zeros((self.rt_nuc.nnuc, 3, S.shape[0], S.shape[0]))

        for i in range(self.rt_nuc.nnuc):
            for j in range(3):
                disp_ar[i,j] = self.displacement

                self.rt_nuc.pos = pos_copy + disp_ar
                mol = self.rt_nuc.get_mol()
                Sf = mol.intor("int1e_ovlp")
               
                self.rt_nuc.pos = pos_copy - disp_ar
                mol = self.rt_nuc.get_mol()
                Si = mol.intor("int1e_ovlp")

                dS_ar[i,j] = (Sf - Si) / (2 * self.displacement)

                disp_ar[i,j] *= 0

        self.rt_nuc.pos = pos_copy
        return dS_ar

    def get_grad_lowdin(self):
        pos_copy = self.rt_nuc.pos
        disp_ar = np.zeros((self.rt_nuc.nnuc, 3))

        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        dV_ar = np.zeros((self.rt_nuc.nnuc, 3, S.shape[0], S.shape[0]))

        #print('----------------------------------------------------')
        #print(np.matmul(fractional_matrix_power(S, 0.5), lowdin(S)))
        #print('----------------------------------------------------')
        #print(np.matmul(np.linalg.inv(lowdin(S)), lowdin(S)))
        #print('----------------------------------------------------')

        for i in range(self.rt_nuc.nnuc):
            for j in range(3):
                disp_ar[i,j] = self.displacement

                self.rt_nuc.pos = pos_copy + disp_ar
                mol = self.rt_nuc.get_mol()
                S = mol.intor("int1e_ovlp")
                Vf = fractional_matrix_power(S, 0.5)
                #Vf = np.linalg.inv(lowdin(S))

                self.rt_nuc.pos = pos_copy - disp_ar
                mol = self.rt_nuc.get_mol()
                S = mol.intor("int1e_ovlp")
                Vi = fractional_matrix_power(S, 0.5)

                dV_ar[i,j] = (Vf - Vi) / (2 * self.displacement)

                disp_ar[i,j] *= 0

        self.rt_nuc.pos = pos_copy
        return dV_ar

    def get_grad_hcore(self):
        pos_copy = self.rt_nuc.pos
        disp_ar = np.zeros((self.rt_nuc.nnuc, 3))

        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        dh_ar = np.zeros((self.rt_nuc.nnuc, 3, S.shape[0], S.shape[0])).astype(np.complex128)

        for i in range(self.rt_nuc.nnuc):
            for j in range(3):
                disp_ar[i,j] = self.displacement

                self.rt_nuc.pos = pos_copy + disp_ar
                mol = self.rt_nuc.get_mol()
                hf = scf.hf.get_hcore(mol)

                self.rt_nuc.pos = pos_copy - disp_ar
                mol = self.rt_nuc.get_mol()
                hi = scf.hf.get_hcore(mol)

                dh_ar[i,j] = (hf - hi) / (2 * self.displacement)

                disp_ar[i,j] *= 0

        self.rt_nuc.pos = pos_copy
        return dh_ar

    def get_grad_veff(self, dms):
        pos_copy = self.rt_nuc.pos
        disp_ar = np.zeros((self.rt_nuc.nnuc, 3))

        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        dG_ar = np.zeros((2, self.rt_nuc.nnuc, 3, S.shape[0], S.shape[0])).astype(np.complex128)

        for i in range(self.rt_nuc.nnuc):
            for j in range(3):
                disp_ar[i,j] = self.displacement

                self.rt_nuc.pos = pos_copy + disp_ar
                mol = self.rt_nuc.get_mol()
                Gaf, Gbf = scf.uhf.get_veff(mol, dms)

                self.rt_nuc.pos = pos_copy - disp_ar
                mol = self.rt_nuc.get_mol()
                Gai, Gbi = scf.uhf.get_veff(mol, dms)

                dG_ar[0,i,j] = (Gaf - Gai) / (2 * self.displacement)
                dG_ar[1,i,j] = (Gbf - Gbi) / (2 * self.displacement)

                disp_ar[i,j] *= 0

        self.rt_nuc.pos = pos_copy
        return dG_ar

    def get_grad_dhP(self, dms):
        dh_ar = self.get_grad_hcore()
        return np.einsum('AXij,sji->AX', dh_ar, dms)

    def get_grad_dGP(self, dms):
        dG_ar = self.get_grad_veff(dms)
        return 0.5 * np.einsum('sAXij,sji->AX', dG_ar, dms)

    def get_grad_FVinvdVP(self, dms):
        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        F = scf.hf.get_hcore(mol) + scf.uhf.get_veff(mol, dms)
        #print('here is fock:')
        #print(F)
        Vinv = lowdin(S)
        dV_ar = self.get_grad_lowdin()
        VinvdV = np.einsum('ij,AXjk->AXik', Vinv, dV_ar)
        dVVinv = np.einsum('AXij,jk->AXik', dV_ar, Vinv)
        FVinvdVP = np.einsum('sij,AXjk,skl->AXil', F, VinvdV, dms)
        PdVVinvF = np.einsum('sij,AXjk,skl->AXil', dms, dVVinv, F)
        return -np.einsum('AXii->AX', FVinvdVP + PdVVinvF)

    def get_grad_elec(self, dms):
        de = self.get_grad_dhP(dms) + self.get_grad_dGP(dms) + self.get_grad_FVinvdVP(dms) 
        return de

    def get_grad_ovlp_analytic(self):
        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        dS = -mol.intor('int1e_ipovlp', comp=3)
        nbasis = S.shape[0]

        atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()

        dS_ar = np.zeros((self.rt_nuc.nnuc, 3, nbasis, nbasis))
        I = np.eye(nbasis)

        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]

            dS_ar[k] = np.einsum('ij,xjl->xil', I[:,p0:p1], dS[:,p0:p1,:]) + np.einsum('xki,kl->xil', dS[:,p0:p1,:], I[p0:p1,:])

        return dS_ar

    def get_grad_lowdin_analytic1(self):
        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        nbasis = S.shape[0]
        dS_ar = self.get_grad_ovlp_analytic()
        e, v = np.linalg.eigh(S)
        dStilde_ar = np.zeros(dS_ar.shape)
        dStilde_unscaled = np.einsum('ia,AXab,bj->AXij', v.T, dS_ar, v)
        for i in range(nbasis):
            for j in range(nbasis):
                dStilde_ar[:,:,i,j] = dStilde_unscaled[:,:,i,j] / (np.sqrt(e[i]) + np.sqrt(e[j]))
        
        dV_ar = np.einsum('ai,AXij,jb->AXab', v, dStilde_ar, v.T)
        
        return dV_ar

    def get_grad_lowdin_analytic2(self):
        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        dS = -mol.intor('int1e_ipovlp', comp=3)
        Vinv = lowdin(S)
        e, v = np.linalg.eigh(S)
        nbasis = e.shape[0]
        etilde = np.zeros((nbasis, nbasis))
        for i in range(nbasis):
            for j in range(nbasis):
                etilde[i,j] = 1. / (np.sqrt(e[i]) + np.sqrt(e[j]))

        atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()

        dV_ar = np.zeros((self.rt_nuc.nnuc, 3, nbasis, nbasis))

        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]

            dMij = np.einsum('ij,ki,xkl,lj->xij', etilde, v[p0:p1], dS[:,p0:p1], v); dMijT = np.einsum('xij->xji', dMij)        

            dV_ar[k] = np.einsum('ij,xjk,kl->xil', v, dMij + dMijT, v.T)

        return dV_ar

    def get_grad_dGP_analytic(self, dms):
        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        nbasis = S.shape[0]
        vj_re, vk_re = grad.rhf.get_jk(mol, dms.real)
        vj_im, vk_im = grad.rhf.get_jk(mol, dms.imag)
        vhf_re = (vj_re[0] + vj_re[1] - vk_re)
        vhf_im = (vj_im[0] + vj_im[1] - vk_im)
        vhf = vhf_re + 1j * vhf_im

        atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()

        dGP_ar = np.zeros((self.rt_nuc.nnuc, 3))

        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]

            dGP_ar[k] = 2 * (np.einsum('sxij,sji->x', vhf.real[:,:,p0:p1,:], dms.real[:,:,p0:p1]) - np.einsum('sxij,sji->x', vhf.imag[:,:,p0:p1,:], dms.imag[:,:,p0:p1]))
        return dGP_ar

    def get_grad_FVinvdVP_analytic(self, dms):
        mol = self.rt_nuc.get_mol()
        S = mol.intor("int1e_ovlp")
        F = scf.hf.get_hcore(mol) + scf.uhf.get_veff(mol, dms)
        nbasis = S.shape[0]
        dS = -mol.intor('int1e_ipovlp', comp=3)
        Vinv = lowdin(S)
        e, v = np.linalg.eigh(S)
        etilde = np.zeros((nbasis, nbasis))
        for i in range(nbasis):
            for j in range(nbasis):
                etilde[i,j] = 1. / (np.sqrt(e[i]) + np.sqrt(e[j]))

        atmlst = range(mol.natm)
        aoslices = mol.aoslice_by_atom()

        FVinvdVP_ar = np.zeros((self.rt_nuc.nnuc, 3))

        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]

            dMij = np.einsum('ij,ki,xkl,lj->xij', etilde, v[p0:p1], dS[:,p0:p1], v); dMijT = np.einsum('xij->xji', dMij)        
            Mat3_1 = np.einsum('sij,jk->sik', F, np.matmul(Vinv, v))
            Mat3_2 = np.einsum('xkl,lm->xkm', dMij + dMijT, v.T)
            Mat3 = np.einsum('sik,xkm->sxim', Mat3_1, Mat3_2)

            FVinvdVP_ar[k] = -np.einsum('sxij,sji->x', Mat3, dms).real * 2

        return FVinvdVP_ar

#class EhrenfestBruteForceOld:
#    def __init__(self, mf,  rt_nuc, displacement=1e-2):
#        self.mf = mf
#        self.rt_nuc = rt_nuc
#        self.displacement = displacement
#
#        nmo = mf.mol.nao_nr()
#        nelec_alpha, nelec_beta = mf.mol.nelec[0], mf.mol.nelec[1]
#        occ_alpha = np.concatenate((np.ones(nelec_alpha), np.zeros(nmo-nelec_alpha)))
#        occ_beta = np.concatenate((np.ones(nelec_beta), np.zeros(nmo-nelec_beta)))
#        # Determine number of matrices: 1 for closed shell/generalized, 2 for open shell
#        if mf.istype('RKS') | mf.istype('RHF'):
#            self.occ = occ_alpha + occ_beta
#        elif mf.istype('UKS') | mf.istype('UHF'):
#            self.occ = np.stack((occ_alpha,occ_beta))
#        elif mf.istype('GKS') | mf.istype('GHF'):
#            self.occ = np.concatenate((np.ones(nelec_alpha+nelec_beta), np.zeros(2*nmo-nelec_alpha-nelec_beta)))
#        else:
#            raise Exception('unknown scf method')
#
#    def get_brute_force(self):
#        den_ao = self.mf.make_rdm1(mo_occ = self.occ)
#        pos_copy = self.rt_nuc.pos
#        E_ar = np.zeros((3, self.rt_nuc.nnuc, 3))
#        E_ar[1,:,:] = self.mf.energy_tot(dm = den_ao) * np.ones((self.rt_nuc.nnuc, 3))
#        disp_ar = np.zeros((self.rt_nuc.nnuc, 3))
#        for i in range(self.rt_nuc.nnuc):
#            for j in range(3):
#                disp_ar[i,j] = self.displacement
#                self.rt_nuc.pos = pos_copy + disp_ar
#                self.mf.mol = self.rt_nuc.get_mol()
#                #print(self.mf.mol.atom)
#                E_ar[0,i,j] = self.mf.energy_tot(dm = den_ao) 
#                self.rt_nuc.pos = pos_copy - disp_ar
#                self.mf.mol = self.rt_nuc.get_mol()
#                #print(self.mf.mol.atom)
#                E_ar[2,i,j] = self.mf.energy_tot(dm = den_ao)
#                disp_ar[i,j] *= 0
#        # D1_arA = (E_ar[0,:,:] - E_ar[1,:,:]) / (self.displacement)
#        # D1_arB = (E_ar[1,:,:] - E_ar[2,:,:]) / (self.displacement)
#        # print(D1_arA - D1_arB)
#        self.rt_nuc.pos = pos_copy
#        return -(E_ar[0,:,:] - E_ar[2,:,:]) / (2 * self.displacement)
#
#    def test_brute_force(self):
#        mo_coeff_copy = self.mf.mo_coeff
#        dm_copy = self.mf.make_rdm1()
#        h1e_copy = self.mf.get_hcore()
#        vhf_copy = self.mf.get_veff(self.mf.mol, dm_copy)
#        pos_copy = self.rt_nuc.pos
#        mol_m = self.mf.mol
#        Em = self.mf.energy_tot()
#        disp_ar = np.zeros((self.rt_nuc.nnuc, 3))
#        disp_ar[0,0] = self.displacement
#        self.rt_nuc.pos = pos_copy + disp_ar
#        mol_f = self.rt_nuc.get_mol()
#        self.mf.mol = mol_f
#        Ef = self.mf.energy_tot() 
#        self.rt_nuc.pos = pos_copy - disp_ar
#        self.mf.mol = self.rt_nuc.get_mol() 
#        Ei = self.mf.energy_tot(dm = dm_copy, h1e = h1e_copy, vhf = vhf_copy) 
#        #print(-(Ef - Em) / (self.displacement))
#        #print(-(Em - Ei) / (self.displacement))
#        self.rt_nuc.pos = pos_copy
#        self.mf.mol = self.rt_nuc.get_mol() 
#        print((Ef, Em, Ei))
#        return -(Ef - Em) / (self.displacement)
#
#
#    def test_brute_force2(self):
#        mo_coeff_copy = self.mf.mo_coeff
#        dm_copy = self.mf.make_rdm1()
#        h1e_copy = self.mf.get_hcore()
#        vhf_copy = self.mf.get_veff(self.mf.mol, dm_copy)
#        pos_copy = self.rt_nuc.pos
#        mol_m = self.mf.mol
#        Em = self.mf.energy_tot()
#        #Em2 =  self.mf.energy_tot(dm = dm_copy, h1e = h1e_copy, vhf = vhf_copy)
#        #self.mf.mol = self.rt_nuc.get_mol()
#        #Em3 = self.mf.energy_tot()
#        #print((Em, Em2, Em3))
#        disp_ar = np.zeros((self.rt_nuc.nnuc, 3))
#        disp_ar[0,0] = self.displacement
#
#        self.rt_nuc.pos = pos_copy + disp_ar
#        mol_f = self.rt_nuc.get_mol()
#        Rfm = translatebasis(mol_f, mol_m)
#        mf_f = scf.UHF(mol_f)
#        mf_f.kernel()
#        print(f'Ef = {mf_f.energy_tot()}')
#        self.mf.mol = mol_f
#        print(np.sum(np.absolute(mf_f.mo_coeff - mo_coeff_copy)))
#        print(np.sum(np.absolute(mf_f.mo_coeff - np.matmul(Rfm, mo_coeff_copy))))
#        self.mf.mo_coeff = mf_f.mo_coeff
#        Ef = self.mf.energy_tot()
#
#        #print(mf_f.mo_coeff)
#        #print(mf_f.make_rdm1())
#        #print(mf_f.get_hcore())
#        #Ef = mf_f.energy_tot() 
#        #Ef = mf_f.energy_tot(dm = dm_copy, h1e = h1e_copy, vhf = vhf_copy) 
#        #Ef = mf_f.energy_tot(h1e = h1e_copy, vhf = vhf_copy) 
#        #Ef = mf_f.energy_tot(dm = np.matmul(Rfm, np.matmul(dm_copy, Rfm.T)))
#        self.rt_nuc.pos = pos_copy - disp_ar
#        mol_i = self.rt_nuc.get_mol()
#        mf_i = scf.UHF(mol_i)
#        mf_i.kernel()
#        #print(mf_i.mo_coeff)
#        #print(mf_f.make_rdm1())
#        #print(mf_i.get_hcore())
#        #Ei = mf_i.energy_tot() 
#        #Ei = mf_i.energy_tot(dm = dm_copy, h1e = h1e_copy, vhf = vhf_copy) 
#        #Ei = mf_i.energy_tot(h1e = h1e_copy, vhf = vhf_copy) 
#        Ei = mf_i.energy_tot(h1e = h1e_copy) 
#        #print(-(Ef - Em) / (self.displacement))
#        #print(-(Em - Ei) / (self.displacement))
#        self.rt_nuc.pos = pos_copy
#        print((Ef, Em, Ei))
#        return -(Ef - Em) / (self.displacement)

