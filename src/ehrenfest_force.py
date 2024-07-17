import numpy
import scipy.linalg
import ctypes
from pyscf import gto, scf, grad
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, _vhf
from pyscf.gto.mole import is_au
from pyscf.lo.orth import lowdin

def grad_lowdin(mol):
    S = mol.intor("int1e_ovlp")
    Vinv = lowdin(S)
    e, v = numpy.linalg.eigh(S)
    nbasis = e.shape[0]
    etilde = numpy.zeros((nbasis, nbasis))
    for i in range(nbasis):
        for j in range(nbasis):
            etilde[i,j] = 1. / (numpy.sqrt(e[i]) + numpy.sqrt(e[j]))
    return etilde, v, Vinv

def get_force(mf_grad):
    grad_nuc = mf_grad.grad_nuc()
    if mf_grad.base.istype('RHF'):
        raise Exception('not implemented')
    elif mf_grad.base.istype('RKS'):
        raise Exception('not implemented')
    elif mf_grad.base.istype('UHF'):
        grad_elec = grad_elec_unrestricted(mf_grad)
    elif mf_grad.base.istype('UKS'):
        raise Exception('not implemented')
    elif mf_grad.base.istype('GHF'):
        raise Exception('not implemented')
    elif mf_grad.base.istype('GKS'):        
        raise Exception('not implemented')
    else:
        raise Exception('unknown scf method')

    return -(grad_nuc + grad_elec)

def get_force_new(mf, dms = None):
    grad_nuc = grad.rhf.grad_nuc(mf.mol)
    grad_elec = grad_elec_new(mf, dms)
    return -(grad_nuc + grad_elec)

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

def grad_elec_new(mf, dms = None):
    '''
    Electronic part of UHF gradients

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''

    mol = mf.mol
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff

    etilde, v, Vinv = grad_lowdin(mol)

    hcore_deriv = hcore_generator_new(mf)
    s1 = -mol.intor('int1e_ipovlp', comp=3)
    if dms is None: dms = mf.make_rdm1(mo_coeff, mo_occ)

    if dms.dtype != numpy.complex128:
        dms = dms.astype(numpy.complex128)

    F = mf.get_fock(dm = dms)

    vj_re, vk_re = grad.rhf.get_jk(mol, dms.real)
    vj_im, vk_im = grad.rhf.get_jk(mol, dms.imag)
    vhf_re = (vj_re[0] + vj_re[1] - vk_re)
    vhf_im = (vj_im[0] + vj_im[1] - vk_im)
    vhf = vhf_re + 1j * vhf_im

    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,sji->x', h1ao, dms).real

        de[k] += (numpy.einsum('sxij,sji->x', vhf.real[:,:,p0:p1,:], dms.real[:,:,p0:p1]) - numpy.einsum('sxij,sji->x', vhf.imag[:,:,p0:p1,:], dms.imag[:,:,p0:p1])) * 2        

        dMij = numpy.einsum('ij,ki,xkl,lj->xij', etilde, v[p0:p1], s1[:,p0:p1], v); dMijT = numpy.einsum('xij->xji', dMij)        
        Mat3_1 = numpy.einsum('sij,jk->sik', F, numpy.matmul(Vinv, v))
        Mat3_2 = numpy.einsum('xkl,lm->xkm', dMij + dMijT, v.T)
        Mat3 = numpy.einsum('sik,xkm->sxim', Mat3_1, Mat3_2)
        de[k] += -numpy.einsum('sxij,sji->x', Mat3, dms).real * 2

    return de


def grad_elec_unrestricted(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of UHF/UKS gradients

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    etilde, v, Vinv = grad_lowdin(mol)

    #dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    #dme0_sf = dme0[0] + dme0[1]

    if mo_coeff.dtype != numpy.complex128:
        mo_coeff = mo_coeff.astype(numpy.complex128)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = mf_grad._tag_rdm1 (dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)

    #F = mf.get_fock()
    F = scf.hf.get_hcore(mol) + scf.uhf.get_veff(mol, dm0)

    #print('start')
    vhf = mf_grad.get_veff(mol, dm0.real) + 1j * mf_grad.get_veff(mol, dm0.imag)
    #print('end')

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]

        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,sji->x', h1ao, dm0).real
        #test_new1 = numpy.einsum('xij,sji->x', h1ao, dm0)
        #print(f'start test 1 k = {k}: {test_new1}')

        de[k] += (numpy.einsum('sxij,sji->x', vhf.real[:,:,p0:p1,:], dm0.real[:,:,p0:p1]) - numpy.einsum('sxij,sji->x', vhf.imag[:,:,p0:p1,:], dm0.imag[:,:,p0:p1])) * 2        

        #test_new2 = (numpy.einsum('sxij,sji->x', vhf.real[:,:,p0:p1,:], dm0.real[:,:,p0:p1]) - numpy.einsum('sxij,sji->x', vhf.imag[:,:,p0:p1,:], dm0.imag[:,:,p0:p1])) * 2
        #print(test_new2)

        dMij = numpy.einsum('ij,ki,xkl,lj->xij', etilde, v[p0:p1], s1[:,p0:p1], v); dMijT = numpy.einsum('xij->xji', dMij)        
        #print('start123')
        Mat3_1 = numpy.einsum('sij,jk->sik', F, numpy.matmul(Vinv, v))
        #print('end1')
        Mat3_2 = numpy.einsum('xkl,lm->xkm', dMij + dMijT, v.T)
        #print('end2')
        Mat3 = numpy.einsum('sik,xkm->sxim', Mat3_1, Mat3_2)
        #print('end3')
        #Mat3_old = numpy.einsum('sij,jk,xkl,lm->sxim', F, numpy.matmul(Vinv, v), dMij + dMijT, v.T)
        #print(Mat3 - Mat3_old)

        de[k] -= numpy.einsum('sxij,sji->x', Mat3, dm0).real * 2

        #test_new3 = numpy.einsum('sxij,sji->x', Mat3, dm0) * 2
        #test_old3 = numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2
        #print(test_new3)
        #print((test_new3 - test_old3)/numpy.absolute(test_old3))

    return de

        #print('-----------------------------------------------------')
        ##print(numpy.einsum('xij,ij->x', h1ao, dm0_sf))
        #TERM1 = numpy.einsum('xij,sji->x', h1ao, dm0).real
        #print(TERM1)
        #print('-----------------------------------------------------')
        ##print('-----------------------------------------------------')
        ##print(numpy.einsum('sxij,sij->x', vhf.real[:,:,p0:p1], dm0.real[:,p0:p1]) * 2)
        #TERM2 = (numpy.einsum('sxij,sji->x', vhf.real[:,:,p0:p1,:], dm0.real[:,:,p0:p1]) + numpy.einsum('sxij,sji->x', vhf.imag[:,:,p0:p1,:], dm0.imag[:,:,p0:p1])) * 2        
        #print(TERM2)
        #print('-----------------------------------------------------')
        ##print('-----------------------------------------------------')
        ##print(numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2)
        #TERM3 = numpy.einsum('sxij,sji->x', Mat3, dm0).real * 2
        #print(TERM3)
        #print('-----------------------------------------------------')

def grad_elec_old(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''
    Electronic part of UHF/UKS gradients

    Args:
        mf_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    #log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    dm0 = mf_grad._tag_rdm1 (dm0, mo_coeff=mo_coeff, mo_occ=mo_occ)

    #t0 = (logger.process_clock(), logger.perf_counter())
    #log.debug('Computing Gradients of NR-UHF Coulomb repulsion')
    vhf = mf_grad.get_veff(mol, dm0)
    #log.timer('gradients of 2e part', *t0)

    dme0 = mf_grad.make_rdm1e(mo_energy, mo_coeff, mo_occ)
    dm0_sf = dm0[0] + dm0[1]
    dme0_sf = dme0[0] + dme0[1]

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm0_sf)
        test_old1 = numpy.einsum('xij,sji->x', h1ao, dm0.real)
        print(f'start test 1 k = {k}: {test_old1}')
# s1, vhf are \nabla <i|h|j>, the nuclear gradients = -\nabla
        
        de[k] += numpy.einsum('sxij,sij->x', vhf[:,:,p0:p1], dm0[:,p0:p1]) * 2
        test_old2= numpy.einsum('sxij,sij->x', vhf[:,:,p0:p1], dm0[:,p0:p1]) * 2
        print(test_old2)

        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2
        test_old3 =  numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0_sf[p0:p1]) * 2
        print(test_old3)

        de[k] += mf_grad.extra_force(ia, locals())

    #if log.verbose >= logger.DEBUG:
    #    log.debug('gradients of electronic part')
    #    rhf_grad._write(log, mol, de, atmlst)
    return de



