import numpy as np
from pyscf import gto, scf, dft
from tides import rt_scf

mol = gto.M(
	verbose = 0,
	atom='Zn 0 0 0')

mf = scf.GHF(mol).x2c()
mf.kernel()

class x2cDeltaField:
    def __init__(self, rt_scf, amplitude, center=0):

        self.amplitude = np.array(amplitude)
        self.center = center
        mol = rt_scf._scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)
        # For X2C property operators must be rotated by the "picture change" associated with 4c -> 2c
        # See https://github.com/pyscf/pyscf/blob/master/examples/x2c/10-picture_change.py
        tdip = -1 * rt_scf._scf.with_x2c.picture_change(('int1e_r_spinor',
                                                       'int1e_sprsp_spinor'))
        self.tdip = tdip

    def calculate_field_energy(self, rt_scf):
        if rt_scf.current_time == self.center:
            return self.amplitude
        else:
            return [0, 0, 0]

    def calculate_potential(self, rt_scf):
        energy = self.calculate_field_energy(rt_scf)
        return -1 * np.einsum('xij,x->ij', self.tdip, energy)

rt_mf = rt_scf.RT_SCF(mf, 0.5, 2000)
rt_mf.observables.update(energy=True, dipole=True)

delta_field = x2cDeltaField(rt_mf, [0.0001, 0.0000, 0.0000])

rt_mf.add_potential(delta_field)

rt_mf.kernel()

