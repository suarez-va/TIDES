import numpy as np
import rt_output

'''
Real-time Observable Functions
'''

def init_observables(rt_mf):
    rt_mf.observables = {
        'energy'  : False,
        'charge'  : False,
        'dipole'  : False,
        'mag'     : False,
        'mo_occ'  : False,
    }

    rt_mf.observables_functions = {
        'energy'  : [get_energy, rt_output.print_energy],
        'charge'  : [get_charge, rt_output.print_charge],
        'dipole'  : [get_dipole, rt_output.print_dipole],
        'mag'     : [get_mag, rt_output.print_mag],
        'mo_occ'  : [get_mo_occ, rt_output.print_mo_occ],
    }


def remove_suppressed_observables(rt_mf):
    if rt_mf.observables['mag']:
        assert rt_mf._scf.istype('GHF') | rt_mf._scf.istype('GKS')

    for key, print_value in rt_mf.observables.items():
        if not print_value:
            del rt_mf.observables_functions[key]

def get_observables(rt_mf):
    for key, function in rt_mf.observables_functions.items():
          function[0](rt_mf, rt_mf.den_ao)

    rt_output.update_output(rt_mf)

def get_energy(rt_mf, den_ao):
    energy = []
    energy.append(rt_mf._scf.energy_tot(dm=den_ao))
    for frag, mask in rt_mf.fragments.items():
        energy.append(frag.energy_tot(dm=den_ao[mask]))

    rt_mf.energy = energy

def get_charge(rt_mf, den_ao):
    # charge = tr(PaoS)
    charge = []
    if rt_mf.nmat == 2:
        charge.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp), axis=0)))
        for frag, mask in rt_mf.fragments.items():
            charge.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp)[mask], axis=0)))
    else:
        charge.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)))
        for frag, mask in rt_mf.fragments.items():
            charge.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)[mask]))
    
    rt_mf.charge = charge

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

    rt_mf.mo_occ = np.diagonal(den_mo)

def get_dipole(rt_mf, den_ao):
    dipole = []
    dipole.append(rt_mf._scf.dip_moment(rt_mf._scf.mol, rt_mf.den_ao,'A.U.', 1))
    for frag, mask in rt_mf.fragments.items():
        dipole.append(frag.dip_moment(frag.mol, den_ao[mask], 'A.U.', 1))
    
    rt_mf.dipole = dipole

def get_mag(rt_mf, den_ao):
    mag = [] 
    Nsp = int(np.shape(rt_mf.ovlp)[0] / 2)
    
    magx = np.sum((den_ao[:Nsp, Nsp:] + den_ao[Nsp:, :Nsp]) * rt_mf.ovlp[:Nsp,:Nsp]) 
    magy = 1j * np.sum((den_ao[:Nsp, Nsp:] - den_ao[Nsp:, :Nsp]) * rt_mf.ovlp[:Nsp,:Nsp])
    magz = np.sum((den_ao[:Nsp, :Nsp] - den_ao[Nsp:, Nsp:]) * rt_mf.ovlp[:Nsp,:Nsp])
    mag.append([magx, magy, magz])

    for frag, mask in rt_mf.fragments.items():
        frag_ovlp = rt_mf.ovlp[mask]
        frag_den_ao = den_ao[mask]
        Nsp = int(np.shape(frag_ovlp)[0] / 2)
    
        magx = np.sum((frag_den_ao[:Nsp, Nsp:] + frag_den_ao[Nsp:, :Nsp]) * frag_ovlp[:Nsp,:Nsp])
        magy = 1j * np.sum((frag_den_ao[:Nsp, Nsp:] - frag_den_ao[Nsp:, :Nsp]) * frag_ovlp[:Nsp,:Nsp])
        magz = np.sum((frag_den_ao[:Nsp, :Nsp] - frag_den_ao[Nsp:, Nsp:]) * frag_ovlp[:Nsp,:Nsp])
        mag.append([magx, magy, magz])
    
    rt_mf.mag = mag
