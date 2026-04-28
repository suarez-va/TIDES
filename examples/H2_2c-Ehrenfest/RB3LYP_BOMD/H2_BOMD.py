from pyscf import gto, scf, md
import numpy as np

# Build mol
mol = gto.M(atom='''
  H    0.0 0.0 0.0
  H    0.0 0.0 0.75
        ''', basis='6-31G')

# Build RKS object
rks = scf.RKS(mol)
rks.xc = 'B3LYP'
rks.conv_check = False
# Run SCF
rks.kernel()

KE_i = 3.625 # 3.625 eV for each H, giving total KE of 7.25 eV
init_velo = np.sqrt(2*(KE_i/27.211386246)/1836.15267343)

myscanner = rks.nuc_grad_method().as_scanner()

myintegrator = md.NVE(myscanner,
                            dt=0.16,
                            steps=26250,
                            veloc=np.array([[0.0, 0.0, -1 * init_velo],
                                           [0.0, 0.0, init_velo]]),
                            data_output="BOMD.md.data",
                            trajectory_output="BOMD.md.xyz").run()

myintegrator.data_output.close()
myintegrator.trajectory_output.close()

