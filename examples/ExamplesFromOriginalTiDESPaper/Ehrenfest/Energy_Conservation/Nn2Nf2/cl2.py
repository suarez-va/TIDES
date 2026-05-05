import os

n_threads = os.environ.get('SLURM_CPUS_PER_TASK', '12')
os.environ['MKL_NUM_THREADS'] = n_threads
os.environ['OPENBLAS_NUM_THREADS'] = n_threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'

from pyscf import gto, dft, lib
from tides import rt_ehrenfest, rt_utils
import numpy as np

lib.num_threads(int(n_threads))

# Build Cl2 molecule
mol = gto.M(
    verbose = 0,
    atom='''
Cl           0.00000000 0.00000000 0.00000000
Cl           0.00000000 0.00000000 2.00000000
''',
    basis='6-31G*', spin=0)

# Build Unrestricted Kohn-Sham object
cl2 = dft.UKS(mol)
cl2.xc = 'B3LYP'
cl2.kernel()

# Create RT_Ehrenfest object
rt_cl2 = rt_ehrenfest.RT_Ehrenfest(cl2, 0.5, 5000, 
        frequency=4, Ne_step=2, N_step=2)

# Set initial velocity
init_eV = 2.0
init_velo = np.sqrt((init_eV/(2*27.2114))*2/63744)
rt_cl2.nuc.vel[0,2] = -1 * init_velo
rt_cl2.nuc.vel[1,2] = init_velo

# Declare which observables to be calculated/printed
rt_cl2.observables.update(energy=True, nuclei=True)

rt_cl2.kernel()
