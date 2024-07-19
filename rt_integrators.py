import numpy as np
from scipy.linalg import expm, inv

'''
Real-time SCF Integrator Functions
'''

def magnus_step(rt_mf):
    '''
    C'(t+dt) = U(t)C'(t-dt)
    U(t) = exp(-i*2dt*F')
    '''

    fock_orth = rt_mf.get_fock_orth(rt_mf.den_ao)
    mo_coeff_orth_old = np.matmul(inv(rt_mf.orth), rt_mf.mo_coeff_old)

    u = expm(-1j*2*rt_mf.timestep*fock_orth)

    mo_coeff_orth_new = np.matmul(u, mo_coeff_orth_old)
    mo_coeff_new = np.matmul(rt_mf.orth, mo_coeff_orth_new)

    rt_mf.mo_coeff_old = rt_mf._scf.mo_coeff
    rt_mf._scf.mo_coeff = mo_coeff_new
    rt_mf.den_ao = rt_mf._scf.make_rdm1(mo_occ=rt_mf.occ)

def magnus_interpol(rt_mf):
    '''
    C'(t+dt) = U(t+0.5dt)C'(t)
    U(t+0.5dt) = exp(-i*dt*F')

    1. Extrapolate F'(t+0.5dt)
    2. Propagate
    3. Build new F'(t+dt), interpolate new F'(t+0.5dt)
    4. Repeat propagation and interpolation until convergence
    '''

    fock_orth = rt_mf.get_fock_orth(rt_mf.den_ao)
    mo_coeff_orth = np.matmul(inv(rt_mf.orth), rt_mf._scf.mo_coeff)
    fock_orth_p12dt = 2 * fock_orth - rt_mf.fock_orth_n12dt

    for iteration in range(rt_mf.magnus_maxiter):
        u = expm(-1j*rt_mf.timestep*fock_orth_p12dt)

        mo_coeff_orth_pdt = np.matmul(u, mo_coeff_orth)
        mo_coeff_ao_pdt = np.matmul(rt_mf.orth, mo_coeff_orth_pdt)

        den_ao_pdt = rt_mf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt,
                                          mo_occ=rt_mf.occ)
        if (iteration > 0 and
        abs(np.linalg.norm(mo_coeff_ao_pdt)
        - np.linalg.norm(mo_coeff_ao_pdt_old)) < rt_mf.magnus_tolerance):

            rt_mf._scf.mo_coeff = mo_coeff_ao_pdt
            rt_mf.den_ao = den_ao_pdt

            rt_mf.fock_orth_n12dt = fock_orth_p12dt
            break

        rt_mf.current_time += rt_mf.timestep
        fock_orth_pdt = rt_mf.get_fock_orth(den_ao_pdt)
        rt_mf.current_time -= rt_mf.timestep

        fock_orth_p12dt = 0.5 * (fock_orth + fock_orth_pdt)

        mo_coeff_ao_pdt_old = mo_coeff_ao_pdt

        rt_mf._scf.mo_coeff = mo_coeff_ao_pdt
        rt_mf.den_ao = den_ao_pdt
    rt_mf.fock_orth_n12dt = fock_orth_p12dt

def rk4(rt_mf):
    '''
    C'(t + dt) = C'(t) + (k1/6 + k2/3 + k3/3 + k4/6)
    dC' = -i * dt * (F'C')
    '''

    fock_orth = rt_mf.get_fock_orth(rt_mf.den_ao)
    mo_coeff_orth = np.matmul(inv(rt_mf.orth), rt_mf._scf.mo_coeff)

    # k1
    k1 = -1j * rt_mf.timestep * (np.matmul(fock_orth,mo_coeff_orth))
    mo_coeff_orth_1 = mo_coeff_orth + 1/2 * k1

    # k2
    k2 = -1j * rt_mf.timestep * (np.matmul(fock_orth,mo_coeff_orth_1))
    mo_coeff_orth_2 = mo_coeff_orth + 1/2 * k2

    # k3
    k3 = -1j * rt_mf.timestep * (np.matmul(fock_orth,mo_coeff_orth_2))
    mo_coeff_orth_3 = mo_coeff_orth + k3

    # k4
    k4 = -1j * rt_mf.timestep * (np.matmul(fock_orth,mo_coeff_orth_3))

    mo_coeff_orth_new = mo_coeff_orth + (k1/6 + k2/3 + k3/3 + k4/6)
    mo_coeff_ao_new = np.matmul(rt_mf.orth, mo_coeff_orth_new)

    rt_mf._scf.mo_coeff = mo_coeff_ao_new
    rt_mf.den_ao = rt_mf._scf.make_rdm1(mo_occ=rt_mf.occ)

INTEGRATORS = {
    'magnus_step' : magnus_step,
    'magnus_interpol' : magnus_interpol,
    'rk4' : rk4,
}

def get_integrator(rt_mf):
    return INTEGRATORS[rt_mf.prop]
