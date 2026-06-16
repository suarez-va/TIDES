import numpy as np
from scipy.linalg import expm
from tides import applyham_pyscf as applyham_pyscf
from tides import fci_mod as fci_mod
import sys

'''
Real-time Integrator Functions
'''

def magnus_step(rt_scf):
    '''
    C'(t+dt) = U(t)C'(t-dt)
    U(t) = exp(-i*2dt*F')
    '''

    fock_orth = rt_scf._fock_orth

    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_scf.update_time()

    u = expm(-1j*2*rt_scf.timestep*fock_orth)

    mo_coeff_orth_new = np.matmul(u, rt_scf.mo_coeff_orth_old)
    
    rt_scf.mo_coeff_orth_old = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    rt_scf._scf.mo_coeff = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_new)
    rt_scf.den_ao = rt_scf._scf.make_rdm1(mo_occ=rt_scf.occ)
    rt_scf._fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)

def magnus_interpol(rt_scf,fs,fc):
    '''
    C'(t+dt) = U(t+0.5dt)C'(t)
    U(t+0.5dt) = exp(-i*dt*F')

    1. Extrapolate F'(t+0.5dt)
    2. Propagate
    3. Build new F'(t+dt), interpolate new F'(t+0.5dt)
    4. Repeat propagation and interpolation until convergence
    '''

    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)
    fock_orth_p12dt = 2 * rt_scf._fock_orth - rt_scf._fock_orth_n12dt
    
    # Update time, mol is updated here if rt_scf is an Ehrenfest obj
    rt_scf.update_time()

    for iteration in range(rt_scf.magnus_maxiter):
        u = expm(-1j*rt_scf.timestep*fock_orth_p12dt)

        mo_coeff_orth_pdt = np.matmul(u, mo_coeff_orth)
        mo_coeff_ao_pdt = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_pdt)
        den_ao_pdt = rt_scf._scf.make_rdm1(mo_coeff=mo_coeff_ao_pdt,
                                          mo_occ=rt_scf.occ)
        rt_scf.current_time += rt_scf.timestep
        fock_orth_pdt = rt_scf.get_fock_orth(den_ao_pdt)
        rt_scf.current_time -= rt_scf.timestep

        if (iteration > 0 and
        abs(np.linalg.norm(mo_coeff_ao_pdt)
        - np.linalg.norm(mo_coeff_ao_pdt_old)) < rt_scf.magnus_tolerance):

            rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
            rt_scf.den_ao = den_ao_pdt
            rt_scf.fock_orth = fock_orth_pdt
            rt_scf.fock_orth_n12dt = fock_orth_p12dt
            break
        fock_orth_p12dt = 0.5 * (rt_scf._fock_orth + fock_orth_pdt)

        mo_coeff_ao_pdt_old = mo_coeff_ao_pdt

        rt_scf._scf.mo_coeff = mo_coeff_ao_pdt
        rt_scf.den_ao = den_ao_pdt
    rt_scf._fock_orth = fock_orth_pdt
    rt_scf._fock_orth_n12dt = fock_orth_p12dt
    diagcorr1RDM = np.real(np.diag(rt_scf.den_ao@rt_scf.ovlp))
    corrdens = np.copy(diagcorr1RDM)
    corrdens = np.insert(corrdens, 0, rt_scf.current_time)
    print(np.real(np.sum(np.diag(rt_scf.den_ao@rt_scf.ovlp))))
    
    np.savetxt(fc, corrdens.reshape(1, corrdens.shape[0]), fs)
    fc.flush()
    sys.stdout.flush()


def rk4(rt_scf,fs,fc):
    '''
    C'(t + dt) = C'(t) + (k1/6 + k2/3 + k3/3 + k4/6)
    dC' = -i * dt * (F'C')
    '''

    fock_orth = rt_scf._fock_orth
    
    # Update time, mol is updated here if rt_scf is Ehrenfest obj
    rt_scf.update_time()

    mo_coeff_orth = rt_scf.rotate_coeff_to_orth(rt_scf._scf.mo_coeff)

    # k1
    k1 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth))
    mo_coeff_orth_1 = mo_coeff_orth + 1/2 * k1

    # k2
    k2 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_1))
    mo_coeff_orth_2 = mo_coeff_orth + 1/2 * k2

    # k3
    k3 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_2))
    mo_coeff_orth_3 = mo_coeff_orth + k3

    # k4
    k4 = -1j * rt_scf.timestep * (np.matmul(fock_orth,mo_coeff_orth_3))

    mo_coeff_orth_new = mo_coeff_orth + (k1/6 + k2/3 + k3/3 + k4/6)
    mo_coeff_ao_new = rt_scf.rotate_coeff_to_ao(mo_coeff_orth_new)

    rt_scf._scf.mo_coeff = mo_coeff_ao_new
    rt_scf.den_ao = rt_scf._scf.make_rdm1(mo_occ=rt_scf.occ)
    rt_scf._fock_orth = rt_scf.get_fock_orth(rt_scf.den_ao)
    den_mo = mo_coeff_ao_new.T @ rt_scf.den_ao @ mo_coeff_ao_new
    diagcorr1RDM = np.real(np.diag(rt_scf.den_ao@rt_scf.ovlp))
    #diagcorr1RDM = np.real(np.diag(den_mo))
    corrdens = np.copy(diagcorr1RDM)
    #corrdens = scf.hf.dip_moment(rt_scf._scf.mol,rt_scf.den_ao)[2]
    corrdens = np.insert(corrdens, 0, rt_scf.current_time)
    #corrdens = np.insert(corrdens, 5, np.sum(diagcorr1RDM))
    
    #print(scf.hf.dip_moment(rt_scf._scf.mol,rt_scf.den_ao))

    np.savetxt(fc, corrdens.reshape(1, corrdens.shape[0]), fs)
    fc.flush()
    sys.stdout.flush()

def rk4cr(rt_cr,fo,fs,fc,eShift):
    '''
    i d/dt|r> = sum(s) X(sr)|s>
    i d/dt C(I) = sum(J) H(JI)C(J)-X(JI)C(J)
    '''
    # Note function f in comments represents derivative equations above

    # Call to update MO coefficient at new time step, for CAS/RAS SCF
    def updateMO(moNew):
        rt_cr.mo_to_ao = np.copy(moNew)
        rt_cr.ao_to_mo = rt_cr.get_ao_to_mo()
        rt_cr._scf.mo_coeff[:,:rt_cr.numP] = np.copy(rt_cr.mo_to_ao)
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()
        rt_cr.mo_to_orth = rt_cr.get_mo_to_orth()
        rt_cr.orth_to_mo = rt_cr.mo_to_orth.conj().T

    # Call to update Hamiltonian at new time step, if Hamiltonian is time-dependent
    def updateHam():
        rt_cr._h1e_orth = rt_cr.get_h1e_orth()
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_orth = rt_cr.get_h2e_orth()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()

    # Collect initial terms
    x_mat = rt_cr.get_x()
    e0, h1Act, h2Act = rt_cr.get_embH(x_mat)
    reci0 = np.copy(rt_cr._scf.ci.real)
    imci0 = np.copy(rt_cr._scf.ci.imag)
    c0 = np.copy(rt_cr._scf.ci)
    mo0 = np.copy(rt_cr.mo_to_ao)
    
    # k1 = f(t0,y0)
    if rt_cr.ras == False:
        ck1 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci0,h1Act,h2Act,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e0-eShift))+(applyham_pyscf.apply_ham_pyscf_check(imci0,h1Act,h2Act,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e0-eShift))
    else:
        ck1 = (-1j*applyham_pyscf.apply_ham_pyscf_complex_ras(reci0,h1Act,h2Act,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e0-eShift,rt_cr.ind))+(applyham_pyscf.apply_ham_pyscf_complex(imci0,h1Act,h2Act,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e0-eShift,rt_cr.ind))
    rk1 = -1j*np.matmul(rt_cr.mo_to_ao,x_mat)

    # c1 and mo1 represent y0 + k1*timestep/2
    c1 = rt_cr._scf.ci + (rt_cr.timestep*ck1/2)
    mo1 = rt_cr.mo_to_ao +(rt_cr.timestep*rk1/2)

    # Update terms to reflect new CI coefficients, new MO coefficients and new time
    if rt_cr._castype == 'CASSCF':
        updateMO(mo1)
    rt_cr.update_time()
    rt_cr._scf.ci = np.copy(c1)
    rt_cr.den_ao = rt_cr.get_den_ao()
    if len(rt_cr._potential) > 0:
        updateHam()

    # Get new equation of motion terms
    x2 = rt_cr.get_x()
    e2, h1a2, h2a2 = rt_cr.get_embH(x2)
    reci1 = np.copy(c1.real)
    imci1 = np.copy(c1.imag)

    # k2 = f(t0 + timestep/2,y0 + k1*timestep/2)
    if rt_cr.ras == False:
        ck2 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci1,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2-eShift))+(applyham_pyscf.apply_ham_pyscf_check(imci1,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2-eShift))
    else:
        ck2 = (-1j*applyham_pyscf.apply_ham_pyscf_complex_ras(reci1,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2-eShift,rt_cr.ind))+(applyham_pyscf.apply_ham_pyscf_complex(imci1,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2-eShift,rt_cr.ind))
    rk2 = -1j*np.matmul(rt_cr.mo_to_ao,x2)

    # c2 and mo2 represent y0 + k2*timestep/2
    c2 = c0 + (rt_cr.timestep*ck2/2)
    mo2 = mo0 + (rt_cr.timestep*rk2/2)

    # Update system. Note time didn't increment but ci coefficients and molecular orbitals are updated
    if rt_cr._castype == 'CASSCF':
        updateMO(mo2)
    rt_cr._scf.ci = np.copy(c2)
    rt_cr.den_ao = rt_cr.get_den_ao()

    # Get new equation of motion terms
    x3 = rt_cr.get_x()
    e3, h1a3, h2a3 = rt_cr.get_embH(x3)
    reci2 = np.copy(c2.real)
    imci2 = np.copy(c2.imag)

    # k3 = f(t0 + timestep/2,y0 + k2*timestep/2)
    if rt_cr.ras == False:
        ck3 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci2,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3-eShift))+(applyham_pyscf.apply_ham_pyscf_check(imci2,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3-eShift))
    else:
        ck3 = (-1j*applyham_pyscf.apply_ham_pyscf_complex_ras(reci2,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3-eShift,rt_cr.ind))+(applyham_pyscf.apply_ham_pyscf_complex(imci2,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3-eShift,rt_cr.ind))
    rk3 = -1j*np.matmul(rt_cr.mo_to_ao,x3)

    # c3 and mo3 represent y0 + k3*timestep
    c3 = c0 + (rt_cr.timestep*ck3)
    mo3 = mo0 + (rt_cr.timestep*rk3)

    # Update system
    rt_cr.update_time()
    if rt_cr._castype == 'CASSCF':
        updateMO(mo3)
    rt_cr._scf.ci = np.copy(c3)
    rt_cr.den_ao = rt_cr.get_den_ao()
    if len(rt_cr._potential) > 0:
        updateHam()

    # Get new equation of motion terms
    x4 = rt_cr.get_x()
    e4, h1a4, h2a4 = rt_cr.get_embH(x4)
    reci3 = np.copy(c3.real)
    imci3 = np.copy(c3.imag)

    # k4 = f(t0 + timestep,y0 + k3*timestep)
    if rt_cr.ras == False:
        ck4 = (-1j*applyham_pyscf.apply_ham_pyscf_check(reci3,h1a4,h2a4,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e4-eShift))+(applyham_pyscf.apply_ham_pyscf_check(imci3,h1a4,h2a4,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e4-eShift))
    else:
        ck4 = (-1j*applyham_pyscf.apply_ham_pyscf_complex_ras(reci3,h1a4,h2a4,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e4-eShift,rt_cr.ind))+(applyham_pyscf.apply_ham_pyscf_complex(imci3,h1a4,h2a4,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e4-eShift,rt_cr.ind))
    rk4 = -1j*np.matmul(rt_cr.mo_to_ao,x4)

    # y1 = (timestep/6)(k1 + 2*k2 + 2*k3 + k4)
    cf = c0 + ((rt_cr.timestep/6)*(ck1+(2*ck2)+(2*ck3)+ck4))
    mof = mo0 + ((rt_cr.timestep/6)*(rk1+(2*rk2)+(2*rk3)+rk4))

    # Update system. Note time doesn't increment
    if rt_cr._castype == 'CASSCF':
        updateMO(mof)
    rt_cr._scf.ci = np.copy(cf)
    #print(cf)
    rt_cr.den_ao = rt_cr.get_den_ao()

    # Collect output file checks
    ef, h1f, h2f = rt_cr.get_embH(np.zeros((rt_cr.numP,rt_cr.numP)))
    output = np.zeros(3)
    output[0] = rt_cr.current_time
    output[1] = fci_mod.get_FCI_E(
                h1f,
                h2f,
                ef,
                cf,
                rt_cr._scf.ncas,
                rt_cr._scf.nelecas[0],
                rt_cr._scf.nelecas[1],
                gen=False,
            )
    output[2] = np.real(np.sum(np.diag(rt_cr.den_ao@rt_cr.ovlp))) # Gives number of electrons. Shouldn't ever change.
    print(output[2])
    
    # corrdens represents AO occupation
    diagcorr1RDM = np.real(np.diag(rt_cr.den_ao@rt_cr.ovlp))
    #print(diagcorr1RDM)
    corrdens = np.copy(diagcorr1RDM)

    corrdens = np.insert(corrdens, 0, rt_cr.current_time)
    
    np.savetxt(fo, output.reshape(1, output.shape[0]), fs)
    fo.flush()
    np.savetxt(fc, corrdens.reshape(1, corrdens.shape[0]), fs)
    fc.flush()
    sys.stdout.flush()

def vv(rt_cr,fo,fs,fc,eShift):
    '''
    Velocity verlet integrator as shown in J. Chem. Theory Comput. 2018, 14, 8, 4129–4138
    This procedure represents equations 4-16
    For TDCASCI only
    '''

    # Update Hamiltonian whenever time increments
    def updateHam():
        rt_cr._h1e_orth = rt_cr.get_h1e_orth()
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_orth = rt_cr.get_h2e_orth()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()

    # Call to update Hamiltonian at new time step
    def updateMO(moNew):
        rt_cr.mo_to_ao = np.copy(moNew)
        rt_cr.ao_to_mo = rt_cr.get_ao_to_mo()
        rt_cr._scf.mo_coeff[:,:rt_cr.numP] = np.copy(rt_cr.mo_to_ao)
        rt_cr._h1e_mo = rt_cr.get_h1e_mo()
        rt_cr._h2e_mo = rt_cr.get_h2e_mo()
        rt_cr.mo_to_orth = rt_cr.get_mo_to_orth()
        rt_cr.orth_to_mo = rt_cr.mo_to_orth.conj().T

    # Initialize terms
    x_mat = rt_cr.get_x()
    qc0 = np.copy(rt_cr._scf.ci.real)
    qm0 = np.copy(rt_cr.mo_to_ao.real)

    if rt_cr.firstStep == True:
        e1, h1a1, h2a1 = rt_cr.get_embH(x_mat)
        pc0 = np.copy(rt_cr._scf.ci.imag)
        pm0 = np.copy(rt_cr.mo_to_ao.imag)
        # Eq 7
        pcDot0 = -applyham_pyscf.apply_ham_pyscf_check(qc0,h1a1,h2a1,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e1-eShift).astype(np.float64)
        pmDot0 = -np.matmul(qm0,x_mat)
        # Eq 8
        pcHalfH = pc0 + (rt_cr.timestep*pcDot0/2)
        pmHalfH = pm0 + (rt_cr.timestep*pmDot0/2)

        qmDot0 = np.matmul(pm0,x_mat)
        qmHalfH = qm0 + (rt_cr.timestep*qmDot0/2)
        qcDot0 = applyham_pyscf.apply_ham_pyscf_check(pc0,h1a1,h2a1,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e1-eShift).astype(np.float64)
        qcHalfH = qc0 + (rt_cr.timestep*qcDot0/2)


    if rt_cr.firstStep == False:
        # Eq 12
        pcHalfH = rt_cr.pcMinusHalf + (rt_cr.timestep*rt_cr.pcDotH)
        pmHalfH = rt_cr.pmMinusHalf + (rt_cr.timestep*rt_cr.pmDotH)

    # Increment Time
    rt_cr.update_time()
    if len(rt_cr._potential) > 0:
        updateHam()
    if rt_cr._castype == 'CASSCF':
        updateMO(qmHalfH + (1j*pmHalfH))
        rt_cr._scf.ci = qcHalfH+(1j*pcHalfH)
        rt_cr.den_ao = rt_cr.get_den_ao()
    x2 = rt_cr.get_x()
    e2, h1a2, h2a2 = rt_cr.get_embH(x2)
    # Eq 9/13
    qcDotHalfH = applyham_pyscf.apply_ham_pyscf_check(pcHalfH,h1a2,h2a2,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e2-eShift).astype(np.float64)
    qmDotHalfH = np.matmul(pmHalfH,x2)
    # Eq 10/14
    qcH = qc0 + (rt_cr.timestep*qcDotHalfH)
    qmH = qm0 + (rt_cr.timestep*qmDotHalfH)

    # Increment Time
    rt_cr.update_time()
    if len(rt_cr._potential) > 0:
        updateHam()
    if rt_cr._castype == 'CASSCF':
        updateMO(qmH+(1j*pmHalfH))
        rt_cr._scf.ci = qcH+(1j*pcHalfH)
        rt_cr.den_ao = rt_cr.get_den_ao()
    x3 = rt_cr.get_x()
    e3, h1a3, h2a3 = rt_cr.get_embH(x3)
    # Eq 11/15
    pcDotH = -applyham_pyscf.apply_ham_pyscf_check(qcH,h1a3,h2a3,rt_cr._scf.nelecas[0],rt_cr._scf.nelecas[1],rt_cr._scf.ncas,e3-eShift).astype(np.float64)
    # Eq 16
    pcH = pcHalfH + (rt_cr.timestep*pcDotH/2)

    # Update system to new timestep
    rt_cr._scf.ci = qcH+(1j*pcH)
    rt_cr.pMinusHalf = np.copy(pcHalfH) # Preps Eq 12 for next step
    rt_cr.pDotH = np.copy(pcDotH) # Preps Eq 12 for next step
    rt_cr.firstStep = False
    rt_cr.den_ao = rt_cr.get_den_ao()

    # Collect output file checks
    ef, h1f, h2f = rt_cr.get_embH(np.zeros((rt_cr.numP,rt_cr.numP)))
    output = np.zeros(3)
    output[0] = rt_cr.current_time
    output[1] = fci_mod.get_FCI_E(
                h1f,
                h2f,
                ef,
                rt_cr._scf.ci,
                rt_cr._scf.ncas,
                rt_cr._scf.nelecas[0],
                rt_cr._scf.nelecas[1],
                gen=False,
            )
    diagcorr1RDM = np.real(np.diag(rt_cr.den_ao@rt_cr.ovlp))
    # corrdens stores AO occupations at the given time step
    corrdens = np.copy(diagcorr1RDM)
    output[2] = np.real(np.sum(np.diag(rt_cr.den_ao@rt_cr.ovlp))) # Gives number of electrons. Shouldn't ever change.
    print(output[2])
    corrdens = np.insert(corrdens, 0, rt_cr.current_time)
    
    np.savetxt(fo, output.reshape(1, output.shape[0]), fs)
    fo.flush()
    np.savetxt(fc, corrdens.reshape(1, corrdens.shape[0]), fs)
    fc.flush()
    sys.stdout.flush()
    

INTEGRATORS = {
    'magnus_step' : magnus_step,
    'magnus_interpol' : magnus_interpol,
    'rk4' : rk4,
    'rk4cr': rk4cr,
    'vv': vv
}

def get_integrator(rt_scf):
    return INTEGRATORS[rt_scf.prop]
