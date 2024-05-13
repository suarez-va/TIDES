# rt_pyscf
Real-Time Electronic Structure Package for PySCF

Currently includes:
  1. Real-time propagation of molecular orbital (MO) coefficient matrix
  2. Supports restricted, unrestricted, and generalized hf/dft methods.
  3. Observables:
        - MO occupations
        - Charge
        - Energy
        - Dipole
        - Magnetization
  4. Propagators:
        - 2nd Order Magnus Step (MMUT)
        - 2nd Order Interpolated Magnus
        - Runge-Kutta 4
  5. Functionality:
        - Localized "noscf" basis
        - Static bfield
        - Delta impulse
        - Excitation
        - Complex Absorbing Potential (CAP)


How to use:
1. Initialize system using hf/dft method of choice in pyscf.
2. Create rt_scf class from the static scf object, enter propagation parameters.
        - rt_scf class must be given the following (mf, timestep, frequency, total_steps, filename)
3. Call the kernel() function to start propagation.

In development:
1. Other pulse excitations
2. Function to generate spectra
3. Nuclear dynamics (?)

Name Ideas:
1. Real-Time Electronic and Nuclear Dynamics (RTEND)
2. PyJK
3. Open Methodological Non-adiabatic Infrastructure (OMNI)
4. Dynamical Electronic Nuclear Surface Hopping Infrastructure (DENSHI)
5. Open-source Package for TIme-dependent CALculations (O.P.TI.CAL.)
