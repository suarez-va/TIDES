import numpy as np
import rt_output

'''
Real-time SCF Observable Functions
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

    rt_mf.time = []
    rt_mf.energy = []
    rt_mf.charge = []
    rt_mf.dipole = []
    rt_mf.mag = []
    rt_mf.mo_occ = []

def remove_suppressed_observables(rt_mf):
    for key, print_value in rt_mf.observables.items():
        if not print_value:
            del rt_mf.observables_functions[key]
            delattr(rt_mf, key)

def get_observables(rt_mf, mo_coeff_print):
    rt_mf.time.append(rt_mf.current_time)
    for key, function in rt_mf.observables_functions.items():
          function[0](rt_mf, rt_mf.den_ao, mo_coeff_print)

    rt_output.update_output(rt_mf)

def get_energy(rt_mf, den_ao, *args):
    energy = []
    energy.append(rt_mf._scf.energy_tot(dm=den_ao))
    for i, mask in enumerate(rt_mf.fragments_indices):
        frag = rt_mf.fragments[i]
        energy.append(frag.energy_tot(dm=den_ao[mask]))

    rt_mf.energy.append(energy)

def get_charge(rt_mf, den_ao, *args):
    # charge = tr(PaoS)
    charge = []
    if rt_mf.nmat == 2:
        charge.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp), axis=0)))
        for _, mask in enumerate(rt_mf.fragments_indices):
            charge.append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp)[mask], axis=0)))
    else:
        charge.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)))
        for _, mask in enumerate(rt_mf.fragments_indices):
            charge.append(np.trace(np.matmul(den_ao,rt_mf.ovlp)[mask]))

    rt_mf.charge.append(charge)

def get_mo_occ(rt_mf, den_ao, mo_coeff_print, *args):
    # P_mo = C+SP_aoSC
    SP_aoS = np.matmul(rt_mf.ovlp,np.matmul(den_ao,rt_mf.ovlp))
    if rt_mf.nmat == 2:
        mo_coeff_print_transpose = np.stack((mo_coeff_print[0].T, mo_coeff_print[1].T))
        den_mo = np.matmul(mo_coeff_print_transpose,np.matmul(SP_aoS,mo_coeff_print))
        den_mo = np.real(np.sum(den_mo,axis=0))
    else:
        den_mo = np.matmul(mo_coeff_print.T, np.matmul(SP_aoS,mo_coeff_print))
        den_mo = np.real(den_mo)

    rt_mf.mo_occ.append(np.diagonal(den_mo))

def get_dipole(rt_mf, den_ao, *args):
    dipole = []
    dipole.append(rt_mf._scf.dip_moment(rt_mf._scf.mol, rt_mf.den_ao,'A.U.', 1))
    for i, mask in enumerate(rt_mf.fragments_indices):
        frag = rt_mf.fragments[i]
        dipole.append(frag.dip_moment(frag.mol, den_ao[mask], 'A.U.', 1))

    rt_mf.dipole.append(dipole)

def get_mag(rt_mf, den_ao, *args):
    mag = []
    Nsp = int(np.shape(rt_mf.ovlp)[0] / 2)

    magx = np.sum((den_ao[:Nsp, Nsp:] + den_ao[Nsp:, :Nsp]) * rt_mf.ovlp[:Nsp,:Nsp])
    magy = 1j * np.sum((den_ao[:Nsp, Nsp:] - den_ao[Nsp:, :Nsp]) * rt_mf.ovlp[:Nsp,:Nsp])
    magz = np.sum((den_ao[:Nsp, :Nsp] - den_ao[Nsp:, Nsp:]) * rt_mf.ovlp[:Nsp,:Nsp])
    mag.append([magx, magy, magz])

    for i, mask in enumerate(rt_mf.fragments_indices):
        frag_ovlp = rt_mf.ovlp[mask]
        frag_den_ao = den_ao[mask]
        Nsp = int(np.shape(frag_ovlp)[0] / 2)

        magx = np.sum((frag_den_ao[:Nsp, Nsp:] + frag_den_ao[Nsp:, :Nsp]) * frag_ovlp[:Nsp,:Nsp])
        magy = 1j * np.sum((frag_den_ao[:Nsp, Nsp:] - frag_den_ao[Nsp:, :Nsp]) * frag_ovlp[:Nsp,:Nsp])
        magz = np.sum((frag_den_ao[:Nsp, :Nsp] - frag_den_ao[Nsp:, Nsp:]) * frag_ovlp[:Nsp,:Nsp])
        mag.append([magx, magy, magz])

    rt_mf.mag.append(mag)
