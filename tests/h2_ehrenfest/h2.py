from pyscf import gto, scf
from tides import rt_ehrenfest, rt_utils
import numpy as np

# Build h2 molecule
h2_mol = gto.M(verbose = 0,
    atom='''
    H           0.00000000 0.00000000 0.00000000
    H           0.00000000 0.00000000 0.8000000
    ''',
    basis='6-31g*', spin=0)

# Build Unrestricted Hartree-Fock object
h2_scf = scf.UKS(h2_mol)
h2_scf.kernel()

# Prepare complex density with unitary rotation between first two MO's
theta = 0.75 * np.pi
phi = 0.49 * np.pi

n = h2_scf.mo_occ
eps = h2_scf.mo_energy
Cao = h2_scf.mo_coeff.astype(np.complex128)

S = h2_scf.get_ovlp()
s, U = np.linalg.eigh(S)
X = np.linalg.multi_dot([U, np.diag(np.power(s, -0.5)), U.T])
Xinv = np.linalg.multi_dot([U, np.diag(np.power(s, 0.5)), U.T])

Coao = np.einsum('ij,sjk->sik', Xinv, Cao)
Coao_excite = np.copy(Coao)
Coao_excite[0,:,0] = np.cos(theta) * np.exp(1j * phi) * Coao[0,:,0] + np.sin(theta) * Coao[0,:,1]
Coao_excite[0,:,1] = -np.sin(theta) * Coao[0,:,0] + np.cos(theta) * np.exp(-1j * phi) * Coao[0,:,1]
Cao_excite = np.einsum('ij,sjk->sik', X, Coao_excite)

h2_scf.mo_coeff = Cao_excite

# Create RT_Ehrenfest object
rt_h2 = rt_ehrenfest.RT_Ehrenfest(h2_scf, 0.005, 10000, prop="magnus_interpol", frequency=250, Ne_step=1, N_step=1)

# Declare which observables to be calculated/printed
rt_h2.observables.update(energy=True, nuclei=True, den_ao=True)

rt_h2.kernel()

