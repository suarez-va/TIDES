import numpy as np

'''
Real-time SCF utilities
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
    # The charge on each fragment will be calculated at every timestep
    nmo = rt_mf._scf.mol.nao_nr()

    rt_mf.fragments = np.zeros((len(fragments),nmo,nmo))

    for index, frag in enumerate(fragments):
        for j, bf in enumerate(rt_mf._scf.mol.ao_labels()):
            if int(bf.split()[0]) in frag:
                rt_mf.fragments[index,j,j] = 1
