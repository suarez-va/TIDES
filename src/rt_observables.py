import numpy as np
import rt_output
from basis_utils import read_mol, mask_fragment_basis
from rt_utils import update_mo_coeff_print
from pyscf import lib
from pyscf.tools import cubegen

'''
Real-time Observable Functions
'''

def _init_observables(rt_mf):
    rt_mf.observables = {
        'energy'       : False,
        'charge'       : False,
        'dipole'       : False,
        'quadrupole'   : False,
        'mag'          : False,
        'mo_occ'       : False,
        'atom_charges' : False,
        'nuclei'       : False,
        'cube_density' : False,
        'mo_coeff'     : False,
        'den_ao'       : False,
        'fock_ao'      : False,
        }

    rt_mf._observables_functions = {
        'energy'       : [get_energy, rt_output._print_energy],
        'charge'       : [get_charge, rt_output._print_charge],
        'dipole'       : [get_dipole, rt_output._print_dipole],
        'quadrupole'   : [get_quadrupole, rt_output._print_quadrupole],
        'mag'          : [get_mag, rt_output._print_mag],
        'mo_occ'       : [get_mo_occ, rt_output._print_mo_occ],
        'atom_charges' : [get_atom_charges, rt_output._print_atom_charges],
        'nuclei'       : [get_nuclei, rt_output._print_nuclei],
        'cube_density' : [get_cube_density, lambda *args: None],
        'mo_coeff'     : [lambda *args: None, rt_output._print_mo_coeff],
        'den_ao'       : [lambda *args: None, rt_output._print_den_ao],
        'fock_ao'      : [lambda *args: None, rt_output._print_fock_ao],
        }



def _remove_suppressed_observables(rt_mf):
    if rt_mf.observables['mag']:
        assert rt_mf._scf.istype('GHF') | rt_mf._scf.istype('GKS')

        
    ### For whatever reason, the dip_moment call for GHF and GKS has arg name 'unit_symbol' instead of 'unit'
    if rt_mf._scf.istype('GHF') | rt_mf._scf.istype('GKS'):
        rt_mf._observables_functions['dipole'][0] = temp_get_dipole
    
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

def temp_get_dipole(rt_mf, den_ao):
    # Temporary fix for argument name discrepancy in GHF.dip_moment ('unit_symbol' instead of 'unit')
    rt_mf._dipole = []
    rt_mf._dipole.append(rt_mf._scf.dip_moment(mol=rt_mf._scf.mol, dm=rt_mf.den_ao, unit_symbol='A.U.', verbose=1))
    for frag in rt_mf.fragments:
        rt_mf._dipole.append(frag.dip_moment(mol=frag.mol, dm=den_ao[frag.mask], unit_symbol='A.U.', verbose=1))

def get_quadrupole(rt_mf, den_ao):
    rt_mf._quadrupole = []
    rt_mf._quadrupole.append(rt_mf._scf.quad_moment(mol=rt_mf._scf.mol, dm=rt_mf.den_ao,unit='A.U.', verbose=1))
    for frag in rt_mf.fragments:
        rt_mf._quadrupole.append(frag.quad_moment(mol=frag.mol, dm=den_ao[frag.mask], unit='A.U.', verbose=1))

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
    
def get_atom_charges(rt_mf, den_ao):
    rt_mf._atom_charges = [[],[]]
    if rt_mf.nmat == 2:
        for i, label in enumerate(rt_mf._scf.mol._atom):
            atom_mask = mask_fragment_basis(rt_mf._scf, [i])
            rt_mf._atom_charges[0].append(label[0])
            rt_mf._atom_charges[1].append(np.trace(np.sum(np.matmul(den_ao,rt_mf.ovlp)[atom_mask], axis=0)))
    else:
        for i, label in enumerate(rt_mf._scf.mol._atom):
            atom_mask = mask_fragment_basis(rt_mf._scf, [i])
            rt_mf._atom_charges[0].append(label[0])
            rt_mf._atom_charges[1].append(np.trace(np.matmul(den_ao,rt_mf.ovlp)[atom_mask]))

def get_nuclei(rt_mf, den_ao):
    rt_mf._nuclei = [rt_mf.nuc.labels, rt_mf.nuc.pos*lib.param.BOHR, rt_mf.nuc.vel*lib.param.BOHR, rt_mf.nuc.force]

def get_cube_density(rt_mf, den_ao):
    '''
    Will create Gaussian cube file for molecule electron density 
    for every propagation time given in rt_mf.cube_density_indices.
    '''
    if np.rint(rt_mf.current_time/rt_mf.timestep) in np.rint(np.array(rt_mf.cube_density_indices)/rt_mf.timestep):
        if hasattr(rt_mf, 'cube_filename'):
            cube_name = f'{rt_mf.cube_filename}{rt_mf.current_time}.cube'
        else:
            cube_name = f'{rt_mf.current_time}.cube'
        cubegen.density(rt_mf._scf.mol, cube_name, den_ao)
