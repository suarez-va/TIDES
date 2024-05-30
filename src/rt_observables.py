import numpy as np
import rt_output
'''
Real-time SCF observables
'''


def get_observables(rt_mf, t, mo_coeff_print):
    ener_tot = get_energy(rt_mf, rt_mf.den_ao)

    dipole = get_dipole(rt_mf, rt_mf.den_ao)
    if rt_mf.nmat == 1:
        den_mo = get_den_mo(rt_mf, rt_mf.den_ao, mo_coeff_print)

        charge = get_charge(rt_mf, rt_mf.den_ao)
    else:
        den_mo_alpha = get_den_mo(rt_mf, rt_mf.den_ao[0], mo_coeff_print[0])
        den_mo_beta = get_den_mo(rt_mf, rt_mf.den_ao[1], mo_coeff_print[1])
        den_mo = den_mo_alpha + den_mo_beta

        charge = get_charge(rt_mf, (rt_mf.den_ao[0]+rt_mf.den_ao[1]))

    if rt_mf.mag:
        mag = get_magnetization(rt_mf)
    else:
        mag = None

    rt_output.update_output_file(rt_mf, t, ener_tot, dipole, den_mo, charge, mag)

def get_energy(rt_mf, den_ao):
    return rt_mf._scf.energy_tot(dm=den_ao)

def get_charge(rt_mf, den_ao):
    # charge = tr(PaoS)
    charge = []
    charge.append(np.trace(np.dot(den_ao,rt_mf.ovlp)))

    for index, mask in enumerate(rt_mf.fragments):
        charge.append(np.trace(np.dot(den_ao,rt_mf.ovlp) * mask))

    return charge

def get_den_mo(rt_mf, den_ao, mo_coeff_print):
    # P_mo = C+SP_aoSC

    SP_aoS = np.dot(rt_mf.ovlp,np.dot(den_ao,rt_mf.ovlp))
    den_mo = np.dot(mo_coeff_print.T,np.dot(SP_aoS,mo_coeff_print))
    return np.real(den_mo)

def get_dipole(rt_mf, den_ao):

    dipole = rt_mf._scf.dip_moment(rt_mf._scf.mol, den_ao, 'A.U.', 1)
    return dipole

def get_magnetization(rt_mf):
    mag = [0,0,0]

    Nsp = int(np.shape(rt_mf.ovlp)[0] / 2)
    for k in range(0, Nsp):
        for j in range(0, Nsp):
            ab_add = rt_mf.den_ao[:Nsp, Nsp:][k,j] + rt_mf.den_ao[Nsp:, :Nsp][k,j]
            mag[0] += ab_add * rt_mf.ovlp[k,j]

            ab_sub = rt_mf.den_ao[:Nsp, Nsp:][k,j] - rt_mf.den_ao[Nsp:, :Nsp][k,j]
            mag[1] += 1j * ab_sub * rt_mf.ovlp[k,j]

            aa_bb = rt_mf.den_ao[:Nsp, :Nsp][k,j] - rt_mf.den_ao[Nsp:, Nsp:][k,j]
            mag[2] += aa_bb * rt_mf.ovlp[k,j]

    return mag
