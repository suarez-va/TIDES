import numpy as np
from pyscf import gto, scf, dft
import rt_scf
from rt_spec import abs_spec
from rt_vapp import ElectricField

mol = gto.M(
	verbose = 0,
	atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  ''',
	basis='6-31G*',
    spin = 0)

mf = dft.RKS(mol)
mf.xc = 'CAMB3LYP' #'PBE0'
mf.kernel()

rt_mf = rt_scf.RT_SCF(mf, 0.02, 100)
rt_mf.observables.update(dipole=True)

class EfieldWithQuad:
    def __init__(self, amplitude, center=0):
        self.amplitude = np.array(amplitude)
        self.center = center

    def calculate_field_energy(self, rt_mf):
        if rt_mf.current_time == self.center:
            return self.amplitude
        else:
            return np.array([0, 0, 0])

    def calculate_potential(self, rt_mf):
        energy = self.calculate_field_energy(rt_mf)
        mol = rt_mf._scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)
        tdip = -1 * mol.intor('int1e_r', comp=3)
        qdip = -1 * mol.intor('int1e_rr', comp=9)
        dipole_term = -1 * np.einsum('xij,x->ij', tdip, energy)
        quadrupole_term = -1 * np.einsum('xij,x->ij', qdip, (energy[:, np.newaxis] * energy[np.newaxis, :]).flatten())
        return dipole_term + quadrupole_term

xdelta_field = EfieldWithQuad([0.0001, 0.0000, 0.0000])
ydelta_field = EfieldWithQuad([0.0000, 0.0001, 0.0000])
zdelta_field = EfieldWithQuad([0.0000, 0.0000, 0.0001])

rt_mf.add_potential(xdelta_field, ydelta_field, zdelta_field)

rt_mf.kernel()

#abs_spec(time, dipole_xyz, 'water_abs_values', 0.0001, 50000, 50) # Zero-pad and exponential damping for clean spectrum
