import numpy as np
from basis_utils import match_fragment_atom, mask_fragment_basis, noscfbasis, read_mol, write_mol
from pyscf import scf

'''
Real-time Utilities
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
        frag.match_indices = match_indices
        rt_mf.fragments[frag] = mask_basis

def update_fragments(rt_mf):
    # Update fragments to new geometry, solve scf problem
    fragments = []
    basis, labels, pos = read_mol(rt_mf._scf.mol)
    for frag, mask in rt_mf.fragments.items():
        frag_indices = frag.match_indices
        delattr(frag, 'match_indices')
        frag_labels = [labels[i] for i in frag_indices]
        frag_pos = [pos[i] for i in frag_indices]
        frag_mol = write_mol(basis, frag_labels, frag_pos)
        frag_mol.verbose = 0
        if hasattr(frag, 'xc'):
            xc = frag.xc
            frag.__init__(frag_mol, xc=xc)
        else:
            frag.__init__(frag_mol)
        frag.kernel() 
        frag.match_indices = frag_indices
        fragments.append(frag)
    rt_mf.fragments = {}
    input_fragments(rt_mf, *fragments)

    # Update mo_coeff_print from new fragmens:
    mf_type = getattr(scf, type(rt_mf._scf).__name__)
    if hasattr(rt_mf._scf, 'xc'):
        xc = rt_mf._scf.xc
        mf_new = mf_type(rt_mf._scf.mol, xc=xc)
    else:
        mf_new = mf_type(rt_mf._scf.mol)
    '''
    if rt_mf._scf.istype('RKS'): mf_new = scf.RKS(rt_mf._scf.mol); mf_new.xc = rt_mf._scf.xc
    elif rt_mf._scf.istype('RHF'): mf_new = scf.RHF(rt_mf._scf.mol)
    elif rt_mf._scf.istype('UKS'): mf_new = scf.UKS(rt_mf._scf.mol); mf_new.xc = rt_mf._scf.xc
    elif rt_mf._scf.istype('UHF'): mf_new = scf.UHF(rt_mf._scf.mol)
    elif rt_mf._scf.istype('GKS'): mf_new = scf.GKS(rt_mf._scf.mol); mf_new.xc = rt_mf._scf.xc
    elif rt_mf._scf.istype('GHF'): mf_new = scf.GHF(rt_mf._scf.mol)
    '''
    mf_new.kernel()
    #rt_mf.mo_coeff_print = noscfbasis(mf_new, *fragments)
    rt_mf.mo_coeff_print = mf_new.mo_coeff

def restart_from_chkfile(rt_mf):
    rt_mf._log.note(f'Restarting from chkfile: {rt_mf.chkfile}.')
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

def print_info(rt_mf, mo_coeff_print):
    rt_mf._log.note('PUT CALC INFO HERE')

    if rt_mf.observables['mo_occ']:
        if mo_coeff_print is None:
            if hasattr(rt_mf, 'mo_coeff_print'):
                pass
            else:
                rt_mf._log.info('mo_coeff_print unspecified. Molecular orbital occupations will be printed in the basis of initial mo_coeff.')

                rt_mf.mo_coeff_print = rt_mf._scf.mo_coeff
    else:
        rt_mf.mo_coeff_print = mo_coeff_print
