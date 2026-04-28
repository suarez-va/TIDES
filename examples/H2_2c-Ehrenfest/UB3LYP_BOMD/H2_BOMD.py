from pyscf import gto, dft, md
import numpy as np

class UKS_asym(dft.uks.UKS):
    def kernel(self, dm0=None, **kwargs):
        udm_guess = np.array([[[ 0.18568439, 0.28537676, 0.00000000, 0.00000000],
                               [ 0.28537676, 0.43859311, 0.00000000, 0.00000000],
                               [ 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                               [ 0.00000000, 0.00000000, 0.00000000, 0.00000000]],
                              [[ 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                               [ 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                               [ 0.00000000, 0.00000000, 0.18568439, 0.28537676],
                               [ 0.00000000, 0.00000000, 0.28537676, 0.43859311]]])
        result = super().kernel(dm0=udm_guess, **kwargs)
        return result

# Build mol
mol = gto.M(atom='''
  H    0.0 0.0 0.0
  H    0.0 0.0 0.75
        ''', basis='6-31G')

# Build UKS object
#uks = scf.UKS(mol)
uks = UKS_asym(mol)
uks.xc = 'B3LYP'
# Run SCF
uks.kernel()

KE_i = 3.625 # 3.625 eV for each H, giving total KE of 7.25 eV
init_velo = np.sqrt(2*(KE_i/27.211386246)/1836.15267343)

myscanner = uks.nuc_grad_method().as_scanner()

myintegrator = md.NVE(myscanner,
                            dt=0.16,
                            steps=26250,
                            veloc=np.array([[0.0, 0.0, -1 * init_velo],
                                           [0.0, 0.0, init_velo]]),
                            data_output="BOMD.md.data",
                            trajectory_output="BOMD.md.xyz").run()

myintegrator.data_output.close()
myintegrator.trajectory_output.close()

