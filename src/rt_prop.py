import numpy as np
import rt_integrators
import rt_observables
import rt_output
from rt_utils import update_chkfile

'''
Real-time Propagation
'''

def propagate(rt_mf, mo_coeff_print):
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

    # Start propagation
    for i in range(0, int(rt_mf.max_time / rt_mf.timestep)):
        if np.mod(i, rt_mf.frequency) == 0:
            rt_observables.get_observables(rt_mf)
            if rt_mf.chkfile is not None:
                update_chkfile(rt_mf)

        rt_mf._integrate_function(rt_mf)

    rt_observables.get_observables(rt_mf)  # Collect observables at final time
    if rt_mf.chkfile is not None:
        update_chkfile(rt_mf)

