import numpy as np
from basis_utils import match_fragment_atom, mask_fragment_basis


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
        rt_mf.fragments.append(frag)
        match_indices = match_fragment_atom(rt_mf._scf, frag)
        mask_basis = mask_fragment_basis(rt_mf._scf, match_indices)
        rt_mf.fragments_indices.append(mask_basis)

def restart_from_chkfile(rt_mf):
    with open(rt_mf.chkfile, 'r') as f:
        chk_lines = f.readlines()
        rt_mf.current_time = np.float64(chk_lines[0].split()[3])
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
