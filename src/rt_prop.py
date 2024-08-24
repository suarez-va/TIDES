import numpy as np
import rt_integrators
import rt_observables
import rt_output
from rt_utils import update_chkfile, update_fragments
from ehrenfest_force import get_force

'''
Real-time Propagation Loops
'''

def scf_propagate(rt_scf, mo_coeff_print):
    prop_init(rt_scf, mo_coeff_print)
    for i in range(0, int(rt_scf.max_time / rt_scf.timestep)):
        if np.mod(i, rt_scf.frequency) == 0:
            rt_observables.get_observables(rt_scf)
            if rt_scf.chkfile is not None:
                update_chkfile(rt_scf)

        rt_scf._integrate_function(rt_scf)
        rt_scf.current_time += rt_scf.timestep

    rt_observables.get_observables(rt_scf)  # Collect observables at final time
    if rt_scf.chkfile is not None:
        update_chkfile(rt_scf)

def ehrenfest_propagate(rt_ehrenfest, mo_coeff_print):
    prop_init(rt_ehrenfest, mo_coeff_print)
    #for i in range(0, self.N_step * self.Ne_step * int(self.max_time / self.timestep)):
    for i in range(0, int(rt_ehrenfest.max_time / rt_ehrenfest.timestep)):
        #if np.mod(i, self.N_step * self.Ne_step * self.frequency) == 0:
        if np.mod(i, rt_ehrenfest.frequency) == 0:
            update_fragments(rt_ehrenfest)
            rt_observables.get_observables(rt_ehrenfest)
            if rt_ehrenfest.chkfile is not None:
                update_chkfile(rt_ehrenfest)

        rt_ehrenfest._integrate_function(rt_ehrenfest)
        rt_ehrenfest.current_time += rt_ehrenfest.timestep
        if np.mod(i, rt_ehrenfest.N_step * rt_ehrenfest.Ne_step) == 0:
            rt_ehrenfest.nuc.get_vel(-0.5 * rt_ehrenfest.N_step * rt_ehrenfest.Ne_step * rt_ehrenfest.timestep)
            rt_ehrenfest.nuc.force = get_force(rt_ehrenfest)
            rt_ehrenfest.nuc.get_vel(-0.5 * rt_ehrenfest.N_step * rt_ehrenfest.Ne_step * rt_ehrenfest.timestep)

    update_fragments(rt_ehrenfest)
    rt_observables.get_observables(rt_ehrenfest)  # Collect observables at final time
    if rt_ehrenfest.chkfile is not None:
        update_chkfile(rt_ehrenfest)


def prop_init(rt_mf, mo_coeff_print):
    if mo_coeff_print is None:
            if hasattr(rt_mf, 'mo_coeff_print'):
                pass
            else:
                rt_mf.mo_coeff_print = rt_mf._scf.mo_coeff
    else:
        rt_mf.mo_coeff_print = mo_coeff_print

    rt_observables._remove_suppressed_observables(rt_mf)

    rt_mf._integrate_function = rt_integrators.get_integrator(rt_mf)
    if rt_mf.prop == "magnus_step":
        rt_mf.mo_coeff_orth_old = rt_mf.rotate_coeff_to_orth(rt_mf._scf.mo_coeff)
    if rt_mf.prop == "magnus_interpol":
        rt_mf._fock_orth = rt_mf.get_fock_orth(rt_mf.den_ao)
        rt_mf._fock_orth_n12dt = rt_mf.get_fock_orth(rt_mf.den_ao)
        if not hasattr(rt_mf, 'magnus_tolerance'): rt_mf.magnus_tolerance = 1e-7
        if not hasattr(rt_mf, 'magnus_maxiter'): rt_mf.magnus_maxiter = 15
