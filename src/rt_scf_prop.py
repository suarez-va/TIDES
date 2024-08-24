import numpy as np
import rt_integrators
import rt_observables
import rt_output
from rt_utils import update_chkfile

'''
Real-time SCF Propagator Loop
'''

def propagate(rt_scf, mo_coeff_print):
    
    if mo_coeff_print is None:
        if hasattr(rt_scf, 'mo_coeff_print'):
            pass
        else:
            rt_scf.mo_coeff_print = rt_scf._scf.mo_coeff
    else:
        rt_scf.mo_coeff_print = mo_coeff_print

    rt_observables.remove_suppressed_observables(rt_scf)

    integrate_function = rt_integrators.get_integrator(rt_scf)
    if rt_scf.prop == "magnus_step":
        rt_scf.mo_coeff_orth_old = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    if rt_scf.prop == "magnus_interpol":
        rt_scf.fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)
        rt_scf.fock_orth_n12dt = rt_scf.get_fock_orth(rt_scf.den_ao)
        if not hasattr(rt_scf, 'magnus_tolerance'): rt_scf.magnus_tolerance = 1e-7
        if not hasattr(rt_scf, 'magnus_maxiter'): rt_scf.magnus_maxiter = 15

    for i in range(0, rt_scf.total_steps):
        if np.mod(i, rt_scf.frequency) == 0:
            rt_observables.get_observables(rt_scf)
            update_chkfile(rt_scf)

        integrate_function(rt_scf)
        rt_scf.current_time += rt_scf.timestep

    rt_observables.get_observables(rt_scf)  # Collect observables at final time
    update_chkfile(rt_scf)
