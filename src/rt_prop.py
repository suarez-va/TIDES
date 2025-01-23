import numpy as np
import rt_integrators
import rt_observables
import rt_output
from rt_utils import update_chkfile, print_info

'''
Real-time Propagation
'''

def propagate(rt_mf, mo_coeff_print):
    print_info(rt_mf, mo_coeff_print)
    rt_observables._check_observables(rt_mf)

    rt_mf._integrate_function = rt_integrators.get_integrator(rt_mf)
    rt_mf._fock_orth = rt_mf.get_fock_orth(rt_mf.den_ao)
    if rt_mf.prop == 'magnus_step':
        rt_mf.mo_coeff_orth_old = rt_mf.rotate_coeff_to_orth(rt_mf._scf.mo_coeff)
    if rt_mf.prop == 'magnus_interpol':
        rt_mf._fock_orth_n12dt = np.copy(rt_mf._fock_orth)
        if not hasattr(rt_mf, 'magnus_tolerance'): rt_mf.magnus_tolerance = 1e-7
        if not hasattr(rt_mf, 'magnus_maxiter'): rt_mf.magnus_maxiter = 15

    # Start propagation
    for i in range(0, int(rt_mf.max_time / rt_mf.timestep)):
        if np.mod(i, rt_mf.frequency) == 0:
            rt_observables.get_observables(rt_mf)
            if rt_mf.chkfile is not None:
                update_chkfile(rt_mf)

        rt_mf._integrate_function(rt_mf)
        if rt_mf.istype('RT_Ehrenfest') and np.mod(int(rt_mf.current_time / rt_mf.timestep - 1), rt_mf.N_step * rt_mf.Ne_step) == rt_mf.N_step * rt_mf.Ne_step -1:
            rt_mf.update_force()

    rt_observables.get_observables(rt_mf)  # Collect observables at final time
    if rt_mf.chkfile is not None:
        update_chkfile(rt_mf)

