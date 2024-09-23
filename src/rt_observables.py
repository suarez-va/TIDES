import numpy as np
import rt_output
from basis_utils import read_mol
from rt_utils import update_mo_coeff_print
from pyscf import lib

'''
Real-time Observable Functions
'''

def _init_observables(rt_mf):
    rt_mf.observables = {
        'energy'  : False,
        'charge'  : False,
        'dipole'  : False,
        'mag'     : False,
        'mo_occ'  : False,
        'nuclei'  : False,
        'mo_coeff': False,
        'den_ao'  : False,
        'fock_ao' : False,
        }

    rt_mf._observables_functions = {
        'energy'  : [get_energy, rt_output._print_energy],
        'charge'  : [get_charge, rt_output._print_charge],
        'dipole'  : [get_dipole, rt_output._print_dipole],
        'mag'     : [get_mag, rt_output._print_mag],
        'mo_occ'  : [get_mo_occ, rt_output._print_mo_occ],
        'nuclei'  : [get_nuclei, rt_output._print_nuclei],
        'mo_coeff': [lambda *args: None, rt_output._print_mo_coeff],
        'den_ao'  : [lambda *args: None, rt_output._print_den_ao],
        'fock_ao' : [lambda *args: None, rt_output._print_fock_ao],
        }



def _remove_suppressed_observables(rt_mf):
    if rt_mf.observables['mag']:
        assert rt_mf._scf.istype('GHF') | rt_mf._scf.istype('GKS')

    for key, print_value in rt_mf.observables.items():
        if not print_value:
            del rt_mf._observables_functions[key]

def get_observables(rt_mf):
    if rt_mf.istype('RT_Ehrenfest') and 'mo_occ' in rt_mf.observables:
        update_mo_coeff_print(rt_mf)
    for key, function in rt_mf._observables_functions.items():
          function[0](rt_mf, rt_mf.den_ao)

    rt_output.update_output(rt_mf)

def get_energy(rt_mf, den_ao):
    rt_mf._energy = []
    rt_mf._energy.append(rt_mf._scf.energy_tot(dm=den_ao))
    if rt_mf.istype('RT_Ehrenfest'):
        ke = rt_mf.nuc.get_ke()
        rt_mf._energy[0] += np.sum(ke)
        rt_mf._kinetic_energy = ke
    for frag in rt_mf.fragments:
        rt_mf._energy.append(frag.energy_tot(dm=den_ao[frag.mask]))
        if rt_mf.istype('RT_Ehrenfest'):
            rt_mf._energy[-1] += np.sum(ke[frag.match_indices])
            

def get_charge(rt_mf, den_ao):
    # charge = tr(PaoS)
    rt_mf._charge = []
    if rt_mf.nmat == 2:
        rt_mf._charge.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp), axis=0)))
        for frag in rt_mf.fragments:
            rt_mf._charge.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp)[frag.mask], axis=0)))
    else:
        rt_mf._charge.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)))
        for frag in rt_mf.fragments:
            rt_mf._charge.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)[frag.mask]))
    

def get_dipole(rt_mf, den_ao):
    rt_mf._dipole = []
    rt_mf._dipole.append(rt_mf._scf.dip_moment(mol=rt_mf._scf.mol, dm=rt_mf.den_ao,unit='A.U.', verbose=1))
    for frag in rt_mf.fragments:
        rt_mf._dipole.append(frag.dip_moment(mol=frag.mol, dm=den_ao[frag.mask], unit='A.U.', verbose=1))
    

def get_mo_occ(rt_mf, den_ao):
    # P_mo = C+SP_aoSC
    SP_aoS = np.matmul(rt_mf.ovlp,np.matmul(den_ao,rt_mf.ovlp))
    if rt_mf.nmat == 2:
        mo_coeff_print_transpose = np.stack((rt_mf.mo_coeff_print[0].T, rt_mf.mo_coeff_print[1].T))
        den_mo = np.matmul(mo_coeff_print_transpose,np.matmul(SP_aoS,rt_mf.mo_coeff_print))
        den_mo = np.real(np.sum(den_mo,axis=0))
    else:
        den_mo = np.matmul(rt_mf.mo_coeff_print.T, np.matmul(SP_aoS,rt_mf.mo_coeff_print))
        den_mo = np.real(den_mo)

    rt_mf._mo_occ = np.diagonal(den_mo)

def get_mag(rt_mf, den_ao):
    rt_mf._mag = [] 
    Nsp = int(np.shape(rt_mf.ovlp)[0] / 2)
    
    magx = np.sum((den_ao[:Nsp, Nsp:] + den_ao[Nsp:, :Nsp]) * rt_mf.ovlp[:Nsp,:Nsp]) 
    magy = 1j * np.sum((den_ao[:Nsp, Nsp:] - den_ao[Nsp:, :Nsp]) * rt_mf.ovlp[:Nsp,:Nsp])
    magz = np.sum((den_ao[:Nsp, :Nsp] - den_ao[Nsp:, Nsp:]) * rt_mf.ovlp[:Nsp,:Nsp])
    rt_mf._mag.append([magx, magy, magz])

    for frag in rt_mf.fragments:
        frag_ovlp = rt_mf.ovlp[frag.mask]
        frag_den_ao = den_ao[frag.mask]
        Nsp = int(np.shape(frag_ovlp)[0] / 2)
    
        magx = np.sum((frag_den_ao[:Nsp, Nsp:] + frag_den_ao[Nsp:, :Nsp]) * frag_ovlp[:Nsp,:Nsp])
        magy = 1j * np.sum((frag_den_ao[:Nsp, Nsp:] - frag_den_ao[Nsp:, :Nsp]) * frag_ovlp[:Nsp,:Nsp])
        magz = np.sum((frag_den_ao[:Nsp, :Nsp] - frag_den_ao[Nsp:, Nsp:]) * frag_ovlp[:Nsp,:Nsp])
        rt_mf._mag.append([magx, magy, magz])
    

def get_nuclei(rt_mf, den_ao):
    rt_mf._nuclei = [rt_mf.nuc.labels, rt_mf.nuc.pos*lib.param.BOHR, rt_mf.nuc.vel*lib.param.BOHR, rt_mf.nuc.force]
