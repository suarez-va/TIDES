from pyscf import gto, scf, lib
from tides import rt_scf
import time
import matplotlib.pyplot as plt
import numpy as np


'''
Test speeds using 1-16 cores
'''
run_times = []
n = np.arange(1, 17, 1)
for i in n:
    lib.num_threads(i)

    start_time = time.time()

    # Build mol
    mol = gto.M(atom='''
      O     0.00000000    -0.00001441    -0.34824012
      H    -0.00000000     0.76001092    -0.93285191
      H     0.00000000    -0.75999650    -0.93290797
            ''', basis='6-31G')
    
    # Build RHF object
    rhf = scf.RHF(mol)

    # Run SCF
    rhf.kernel()

    # Declare propagation parameters
    rt_rhf = rt_scf.RT_SCF(rhf, 
    timestep=1.0, max_time=1000, verbose=0)
    
    # Collect energy (won't print since verbose=0)
    rt_rhf.observables['energy'] = True
    rt_rhf.kernel()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    run_times.append(elapsed_time)
    print(f'N = {i}: {elapsed_time:.4f} seconds')
    print(f'{rt_rhf._energy}')

plt.figure()
plt.scatter(n, run_times)
plt.ylabel('Time (seconds)')
plt.xlabel('N')
plt.savefig('Speed_Times.png', bbox_inches='tight')
