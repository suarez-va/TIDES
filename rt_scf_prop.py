import numpy as np
import rt_integrators
import rt_observables
import rt_output
from rt_utils import update_chkfile

'''
Real-time SCF Propagator Loop
'''

def propagate(rt_mf, mo_coeff_print):
    rt_observables.remove_suppressed_observables(rt_mf)

    integrate_function = rt_integrators.get_integrator(rt_mf)
    if rt_mf.prop == "magnus_step":
        rt_mf.mo_coeff_old = rt_mf._scf.mo_coeff
    if rt_mf.prop == "magnus_interpol":
        rt_mf.fock_orth_n12dt = rt_mf.get_fock_orth(rt_mf.den_ao)
        if not hasattr(rt_mf, 'magnus_tolerance'): rt_mf.magnus_tolerance = 1e-7
        if not hasattr(rt_mf, 'magnus_maxiter'): rt_mf.magnus_maxiter = 15

    for i in range(0, rt_mf.total_steps):
        if np.mod(i, rt_mf.frequency) == 0:
            rt_observables.get_observables(rt_mf, mo_coeff_print)
            update_chkfile(rt_mf)

        integrate_function(rt_mf)
        rt_mf.current_time += rt_mf.timestep

    rt_observables.get_observables(rt_mf, mo_coeff_print)  # Collect observables at final time
    update_chkfile(rt_mf)
