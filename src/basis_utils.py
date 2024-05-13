import numpy as np
from pyscf.tools import molden
from pyscf.lo.orth import orth_ao
import scipy

'''
Basis utility functions
'''

def print2molden(mf, filename=None, mo_coeff=None):
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if filename is None: filename = 'file'
    with open(filename + '.molden', 'w') as moldenfile:
        molden.header(mf.mol, moldenfile)
        if len(np.shape(mo_coeff)) < 3:
            molden.orbital_coeff(mf.mol, moldenfile, mo_coeff,
                                 ene=mf.mo_energy, occ=mf.mo_occ)
        else:
            molden.orbital_coeff(mf.mol, moldenfile, mo_coeff[0],
                                 ene=mf.mo_energy[0], occ=mf.mo_occ[0], spin = 'Alpha')
            molden.orbital_coeff(mf.mol, moldenfile, mo_coeff[1],
                                 ene=mf.mo_energy[1], occ=mf.mo_occ[1], spin = 'Beta')


def noscfbasis(mf, *fragments):
    noscf_orbitals = np.zeros(np.shape(mf.mo_coeff))
    ao = int(mf.mol.nao_nr())
    old_orbitals = mf.mo_coeff
    if len(np.shape(mf.mo_coeff)) < 3:
        orb_iteration = 0
        ao_iteration = 0
        # Fill in occupied orbitals
        for frag in fragments:
            occ_per_frag = max(frag.mol.nelec)
            ao_per_frag = frag.mol.nao_nr()
            # Create indexing ranges
            orb_start = orb_iteration
            orb_end = occ_per_frag + orb_iteration
            ao_start = ao_iteration
            ao_end = ao_per_frag + ao_iteration
            noscf_orbitals[ao_start:ao_end,orb_start:orb_end] = frag.mo_coeff[:,:occ_per_frag]
            orb_iteration += occ_per_frag
            ao_iteration += ao_per_frag

        virtual_start = orb_iteration
        # Fill in virtual orbitals
        orb_iteration = 0
        ao_iteration = 0
        for frag in fragments:
            occ_per_frag = max(frag.mol.nelec)
            ao_per_frag = frag.mol.nao_nr()
            # Create indexing ranges
            orb_start = virtual_start + orb_iteration
            orb_end = orb_start + (ao_per_frag - occ_per_frag)
            ao_start = 0 + ao_iteration
            ao_end = ao_per_frag + ao_iteration
            noscf_orbitals[ao_start:ao_end,orb_start:orb_end] = frag.mo_coeff[:,occ_per_frag::]
            orb_iteration += (ao_per_frag - occ_per_frag)
            ao_iteration += ao_per_frag
    else:
        orb_iteration_alpha = 0
        orb_iteration_beta = 0
        ao_iteration = 0
        # Fill in occupied orbitals
        for frag in fragments:
            occ_per_frag_alpha = (frag.mol.nelec[0])
            occ_per_frag_beta = (frag.mol.nelec[1])
            ao_per_frag = frag.mol.nao_nr()
            # Create indexing ranges
            orb_start_alpha = orb_iteration_alpha
            orb_end_alpha = occ_per_frag_alpha + orb_iteration_alpha
            orb_start_beta = orb_iteration_beta
            orb_end_beta = occ_per_frag_beta + orb_iteration_beta
            ao_start = ao_iteration
            ao_end = ao_per_frag + ao_iteration
            noscf_orbitals[0,ao_start:ao_end, orb_start_alpha:orb_end_alpha] = frag.mo_coeff[0,:, :occ_per_frag_alpha]
            noscf_orbitals[1,ao_start:ao_end, orb_start_beta:orb_end_beta] = frag.mo_coeff[1,:, :occ_per_frag_beta]
            orb_iteration_alpha += occ_per_frag_alpha
            orb_iteration_beta += occ_per_frag_beta
            ao_iteration += ao_per_frag
        virtual_start_alpha = orb_iteration_alpha
        virtual_start_beta = orb_iteration_beta
        # Fill in virtual orbitals
        orb_iteration_alpha = 0
        orb_iteration_beta = 0
        ao_iteration = 0
        for frag in fragments:
            occ_per_frag_alpha = (frag.mol.nelec[0])
            occ_per_frag_beta = (frag.mol.nelec[1])
            ao_per_frag = frag.mol.nao_nr()
            # Create indexing ranges
            orb_start_alpha = virtual_start_alpha + orb_iteration_alpha
            orb_end_alpha = orb_start_alpha + (ao_per_frag - occ_per_frag_alpha)
            orb_start_beta = virtual_start_beta + orb_iteration_beta
            orb_end_beta = orb_start_beta + (ao_per_frag - occ_per_frag_beta)
            ao_start = 0 + ao_iteration
            ao_end = ao_per_frag + ao_iteration
            noscf_orbitals[0,ao_start:ao_end, orb_start_alpha:orb_end_alpha] = frag.mo_coeff[0,:, occ_per_frag_alpha::]
            noscf_orbitals[1,ao_start:ao_end, orb_start_beta:orb_end_beta] = frag.mo_coeff[1,:, occ_per_frag_beta::]
            orb_iteration_alpha += (ao_per_frag - occ_per_frag_alpha)
            orb_iteration_beta += (ao_per_frag - occ_per_frag_beta)
            ao_iteration += ao_per_frag

    mf.mo_coeff = noscf_orbitals
    # Orthogonalize
    orth_ao(mf)
    noscf_orbitals = mf.mo_coeff
    mf.mo_coeff = old_orbitals
    return noscf_orbitals
