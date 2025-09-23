import numpy as np
from pyscf import gto, scf, dft
from tides.rt_scf import RT_SCF
from pyscf.scf.stability import ghf_stability ##
from tides.staticfield import static_bfield

mag_y = 0.000085 # in au

HTrimerMol = gto.M(
	verbose = 0,
	atom='''
 H                 0.00000000    0.0 0.909326674
 H                 -1.05    0.0 -0.909326674
 H                 1.05    0.0 -0.909326674
''',
	basis='3-21G',
    spin = 1)


HTrimer = scf.GHF(HTrimerMol)
HTrimer.kernel()

# Run stability analysis to verify our initial state
stable = False
while not stable:
    mo_new, stable = ghf_stability(HTrimer, return_status=True, tol=1e-8)
    dm1 = HTrimer.make_rdm1(mo_new, HTrimer.mo_occ)
    HTrimer.kernel(dm1.astype(np.complex128))


# Add BField, create RT_SCF
static_bfield(HTrimer, [0, mag_y, 0])
rt_mf = RT_SCF(HTrimer, 0.5, 82000)

rt_mf.observables.update(mag=True, hirsh_mag=True)

# Run
rt_mf.kernel()

