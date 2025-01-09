import numpy as np
from pyscf import gto, scf, dft, data
from pyscf.scf.atom_ks import get_atm_nrks
from pyscf.dft import numint
from pyscf.hirshfeld import HirshfeldAnalysis

def hirshfeld_partition(mf, den_ao, grids=None, atom_weights=None):
    if grids is None or atom_weights is None: grids, atom_weights = get_weights(mf.mol)
    den_ao = den_ao.astype(np.complex128)
    
    # Restricted
    if mf.istype('RHF') | mf.istype('RKS'):
        rho = cast_den_ao(mf, den_ao, grids)
        return rho * atom_weights
    # Unrestricted - break into aa, bb spin blocks
    if mf.istype('UHF') | mf.istype('UKS'):
        rho_a = cast_den_ao(mf, den_ao[0], grids)
        rho_b = cast_den_ao(mf, den_ao[1], grids)
        return rho_a * atom_weights, rho_b * atom_weights
    # Generalized - break into aa, ab, ba, bb spin blocks
    if mf.istype('GHF') | mf.istype('GKS'):
        ovlp = mf.get_ovlp()
        Nsp = int(ovlp.shape[0]/2)

        rho_aa = cast_den_ao(mf, den_ao[:Nsp, :Nsp], grids)
        rho_ab = cast_den_ao(mf, den_ao[:Nsp, Nsp:], grids)
        rho_ba = cast_den_ao(mf, den_ao[Nsp:, :Nsp], grids)
        rho_bb = cast_den_ao(mf, den_ao[Nsp:, Nsp:], grids)
        return rho_aa * atom_weights, rho_ab * atom_weights, rho_ba * atom_weights, rho_bb * atom_weights

def get_mag(rho_aa, rho_ab, rho_ba, rho_bb):
    mx = (rho_ab + rho_ba)
    my = 1j * (rho_ab - rho_ba)
    mz = (rho_aa - rho_bb)
    return mx, my, mz

def cast_den_ao(mf, den_ao, grids):
    # Re Part
    rho_re = numint.get_rho(numint.NumInt(), mf.mol, np.real(den_ao), grids)
    # Imag Part
    rho_imag = numint.get_rho(numint.NumInt(), mf.mol, np.imag(den_ao), grids)

    return (rho_re + 1j * rho_imag) * grids.weights

def get_weights(mol):
    grids = dft.Grids(mol)
    grids.build()
    mf = dft.RKS(mol)
    mf.xc = 'HF'
    mf.grids = grids
    mf.kernel()

    H = HirshfeldAnalysis(mf).run()
    return grids, H.result['weights_free']

'''
def OLDget_weights(mf):

    # First build grid for mol
    grids = dft.Grids(mf.mol)
    grids.build()

    # Calculate sph atomic densities in AO basis
    atom_mols, atom_dms = atomic_dm(mf.mol)

    # Cast atomic densities from AO basis to grid
    atom_sph_rhos = []
    for idx, atom_dm in enumerate(atom_dms):
        ao = numint.eval_ao(atom_mols[idx], grids.coords)
        atom_sph_rho = numint.eval_rho(atom_mols[idx], ao, atom_dm)
        atom_sph_rhos.append(atom_sph_rho)

        #atom_sph_rhos.append(numint.get_rho(numint.NumInt(), atom_mols[idx], atom_dm, grids))

    # Calculate promolecular density rho_promolecule = \sum{rho_atomic}
    pro_rho = np.sum(atom_sph_rhos, axis=0)

    # Calculate weight functions for each atom
    atom_weights = []
    for atom_sph_rho in atom_sph_rhos:
        atom_weight = atom_sph_rho / pro_rho
        np.nan_to_num(atom_weight, copy=False)
        atom_weights.append(atom_weight)
    return grids, atom_weights

def atomic_dm(mol, basis='augccpvqz'):
    atom_dms = []
    atom_mols = []
    atom_results = get_atm_nrks(mol)    
    for atom in mol._atom:
        atom_result = atom_results[atom[0]]
        mo_coeff = atom_result[2]
        mo_occ = atom_result[3]
        mocc = mo_coeff[:,mo_occ>0]
        atom_dms.append((mocc*mo_occ[mo_occ>0]).dot(mocc.conj().T))
    return atom_mols, atom_dms
'''
