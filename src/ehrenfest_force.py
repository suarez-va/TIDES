import numpy
import ctypes
from pyscf import gto, scf, grad
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf.gto.mole import is_au
#from pyscf.lo.orth import lowdin
from scipy.linalg import fractional_matrix_power

def get_force(mf, den_ao):
    grad_nuc = grad.rhf.grad_nuc(mf.mol)
    if mf.istype('RHF') | mf.istype('RKS'):
        grad_elec = grad_elec_restricted(mf, den_ao)
    elif mf.istype('UHF') | mf.istype('UKS'):
        grad_elec = grad_elec_unrestricted(mf, den_ao)
    elif mf.istype('GHF') | mf.istype('GKS'):
        #grad_elec = grad_elec_generalized(mf, den_ao)
        raise Exception('not implemented')
    return -(grad_nuc + grad_elec)

def grad_lowdin(mol):
    S = mol.intor("int1e_ovlp")
    #Vinv = lowdin(S)
    Vinv = fractional_matrix_power(S, -0.5)
    e, v = numpy.linalg.eigh(S)
    nbasis = e.shape[0]
    etilde = numpy.zeros((nbasis, nbasis))
    for i in range(nbasis):
        for j in range(nbasis):
            etilde[i,j] = 1. / (numpy.sqrt(e[i]) + numpy.sqrt(e[j]))
    return etilde, v, Vinv

def hcore_generator_new(mf):
    mol = mf.mol
    aoslices = mol.aoslice_by_atom()
    h1 = grad.rhf.get_hcore(mol)
    def hcore_deriv(atm_id):
        shl0, shl1, p0, p1 = aoslices[atm_id]
        with mol.with_rinv_at_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= -mol.atom_charge(atm_id)
            vrinv[:,p0:p1] += h1[:,p0:p1]
            return vrinv + vrinv.transpose(0,2,1)
    return hcore_deriv

def grad_elec_restricted(mf, den_ao):
    '''
    Electronic part of RHF/RKS gradients
    '''

    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf. mo_occ

    etilde, v, Vinv = grad_lowdin(mol)
    hcore_deriv = hcore_generator_new(mf)
    dS = -mol.intor('int1e_ipovlp', comp=3)

    if den_ao is None: den_ao = mf.make_rdm1(mo_coeff, mo_occ)
    if den_ao.dtype != numpy.complex128:
        den_ao = den_ao.astype(numpy.complex128)

    fock = mf.get_fock(dm = den_ao)

    vj_re, vk_re = grad.rhf.get_jk(mol, den_ao.real)
    vj_im, vk_im = grad.rhf.get_jk(mol, den_ao.imag)
    vj = vj_re + 1j * vj_im
    vk = vk_re + 1j * vk_im

    vhf = vj - 0.5 * vk

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, den_ao).real

        de[k] += (numpy.einsum('xij,ji->x', vhf.real[:,p0:p1,:], den_ao.real[:,p0:p1]) - numpy.einsum('xij,ji->x', vhf.imag[:,p0:p1,:], den_ao.imag[:,p0:p1])) * 2        

        dStilde_bra = numpy.einsum('ji,xjk,kl->xil', v[p0:p1], dS[:,p0:p1], v); dStilde_ket = numpy.einsum('xij->xji', dStilde_bra)
        dStilde = dStilde_bra + dStilde_ket
        dVtilde = numpy.einsum('ij,xij->xij', etilde, dStilde)
        dV = numpy.einsum('ij,xjk,lk->xil', v, dVtilde, v)
        VinvdV = numpy.einsum('ij,xjk->xik', Vinv, dV)
        PF = numpy.einsum('ij,jk->ik', den_ao, fock)
        de[k] += -2 * numpy.einsum('xij,ji->x', VinvdV, PF).real

    return de

def grad_elec_unrestricted(mf, den_ao):
    '''
    Electronic part of UHF/UKS gradients
    '''

    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_occ = mf. mo_occ

    etilde, v, Vinv = grad_lowdin(mol)
    hcore_deriv = hcore_generator_new(mf)
    dS = -mol.intor('int1e_ipovlp', comp=3)

    if den_ao is None: den_ao = mf.make_rdm1(mo_coeff, mo_occ)
    if den_ao.dtype != numpy.complex128:
        den_ao = den_ao.astype(numpy.complex128)

    fock = mf.get_fock(dm = den_ao)

    vj_re, vk_re = grad.rhf.get_jk(mol, den_ao.real)
    vj_im, vk_im = grad.rhf.get_jk(mol, den_ao.imag)
    vj = vj_re + 1j * vj_im
    vk = vk_re + 1j * vk_im

    vhf = vj[0] + vj[1] - vk

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,sji->x', h1ao, den_ao).real

        de[k] += (numpy.einsum('sxij,sji->x', vhf.real[:,:,p0:p1,:], den_ao.real[:,:,p0:p1]) - numpy.einsum('sxij,sji->x', vhf.imag[:,:,p0:p1,:], den_ao.imag[:,:,p0:p1])) * 2

        dStilde_bra = numpy.einsum('ji,xjk,kl->xil', v[p0:p1], dS[:,p0:p1], v); dStilde_ket = numpy.einsum('xij->xji', dStilde_bra)
        dStilde = dStilde_bra + dStilde_ket
        dVtilde = numpy.einsum('ij,xij->xij', etilde, dStilde)
        dV = numpy.einsum('ij,xjk,lk->xil', v, dVtilde, v)
        VinvdV = numpy.einsum('ij,xjk->xik', Vinv, dV)
        PF = numpy.einsum('sij,sjk->ik', den_ao, fock)
        de[k] += -2 * numpy.einsum('xij,ji->x', VinvdV, PF).real

    return de


