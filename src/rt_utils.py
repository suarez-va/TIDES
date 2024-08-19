import numpy as np
from basis_utils import match_fragment_atom, mask_fragment_basis, noscfbasis, read_mol, write_mol
from pyscf import scf

'''
Real-time SCF Utilities
'''

def excite(rt_mf, excitation_alpha=None, excitation_beta=None):
    # Excite an electron from the index specified
    if rt_mf.nmat == 1:
        excitation = excitation_alpha
        rt_mf.occ[excitation-1] -= 1
    else:
        if excitation_alpha:
            rt_mf.occ[0][excitation_alpha-1] -= 1
        if excitation_beta:
            rt_mf.occ[1][excitation_beta-1] -= 1

    rt_mf.den_ao = rt_mf._scf.make_rdm1(mo_occ = rt_mf.occ)

def input_fragments(rt_mf, *fragments):
    # Specify the relevant atom indices for each fragment
    # The charge, energy, dipole, and magnetization on each fragment
    # can be calculated

    nmo = rt_mf._scf.mol.nao_nr()
    for index, frag in enumerate(fragments):
        match_indices = match_fragment_atom(rt_mf._scf, frag)
        mask_basis = mask_fragment_basis(rt_mf._scf, match_indices)
        rt_mf.fragments[frag] = mask_basis

def update_fragments(rt_mf, match_indices_array=None):
    if match_indices_array is not None:
        rt_mf.fragments = {}
        fragments = []
        basis, labels, pos = read_mol(rt_mf._scf.mol)
        for index, match_indices in enumerate(match_indices_array):
            frag_labels = [labels[i] for i in match_indices]
            frag_pos = [pos[i] for i in match_indices]
            frag_mol = write_mol(basis, frag_labels, frag_pos)
            if rt_mf._scf.istype('RHF'): frag = scf.RHF(frag_mol)
            elif rt_mf._scf.istype('RKS'): frag = scf.RKS(frag_mol)
            elif rt_mf._scf.istype('UHF'): frag = scf.UHF(frag_mol)
            elif rt_mf._scf.istype('UKS'): frag = scf.UKS(frag_mol)
            elif rt_mf._scf.istype('GHF'): frag = scf.GHF(frag_mol)
            elif rt_mf._scf.istype('GKS'): frag = scf.GKS(frag_mol)
            frag.kernel() 
            fragments.append(frag)
        input_fragments(rt_mf, *fragments)
        if rt_mf._scf.istype('RHF'): mf_scf = scf.RHF(rt_mf_scf.mol)
        elif rt_mf._scf.istype('RKS'): mf_scf = scf.RKS(rt_mf._scf.mol)
        elif rt_mf._scf.istype('UHF'): mf_scf = scf.UHF(rt_mf._scf.mol)
        elif rt_mf._scf.istype('UKS'): mf_scf = scf.UKS(rt_mf._scf.mol)
        elif rt_mf._scf.istype('GHF'): mf_scf = scf.GHF(rt_mf._scf.mol)
        elif rt_mf._scf.istype('GKS'): mf_scf = scf.GKS(rt_mf._scf.mol)
        mf_scf.kernel()
        #mo_coeff_print = noscfbasis(mf_scf, *fragments)
        mo_coeff_print = mf_scf.mo_coeff
    else:
        mo_coeff_print = None
    return mo_coeff_print

def restart_from_chkfile(rt_mf):
    with open(rt_mf.chkfile, 'r') as f:
        chk_lines = f.readlines()
        rt_mf.current_time = float(chk_lines[0].split()[3])
        if rt_mf.nmat == 1:
            rt_mf._scf.mo_coeff = np.loadtxt(chk_lines[2:], dtype=np.complex128)
        else:
            for i, line in enumerate(chk_lines):
                if "Beta" in line:
                    b0 = i
                    break

            mo_alpha0 = np.loadtxt(chk_lines[3:b0], dtype=np.complex128)
            mo_beta0 = np.loadtxt(chk_lines[b0+1:], dtype=np.complex128)
            rt_mf._scf.mo_coeff = np.stack((mo_alpha0, mo_beta0))

def update_chkfile(rt_mf):
    with open(rt_mf.chkfile, 'w') as f:
        f.write(f"Current Time (AU): {rt_mf.current_time} \nMO Coeffs: \n")
        if rt_mf.nmat == 1:
            np.savetxt(f, rt_mf._scf.mo_coeff)
        else:
            f.write("Alpha \n")
            np.savetxt(f, rt_mf._scf.mo_coeff[0])
            f.write("Beta \n")
            np.savetxt(f, rt_mf._scf.mo_coeff[1])
