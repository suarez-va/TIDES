import numpy as np
from pyscf import gto, scf, dft
import rt_scf
import rt_vapp
from rt_vapp import ElectricField

############################
# CURRENTLY DOESNT WORK    #
# PYSCF HAS BUG WITH       #
# dip_moment() for GHF+X2C #
############################

mag_z = 0.000085 # in au

mol = gto.M(
	verbose = 0,
	atom='H 0 0 0',
	basis='STO-3G',
    spin = 1)

mf = scf.ghf.GHF(mol).x2c()
mf.kernel()
print(mf.mo_energy)

class deltaField:
    def __init__(self, rt_mf, amplitude, center=0):

        self.amplitude = np.array(amplitude)
        self.center = center
        mol = rt_mf._scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)
        tdip = -1 * rt_mf._scf.with_x2c.picture_change(('int1e_r_spinor',
                                                       'int1e_sprsp_spinor'))
        self.tdip = tdip

    def calculate_field_energy(self, rt_mf):
        if rt_mf.current_time == self.center:
            return self.amplitude
        else:
            return [0, 0, 0]

    def calculate_potential(self, rt_mf):
        energy = self.calculate_field_energy(rt_mf)
        return -1 * np.einsum('xij,x->ij', self.tdip, energy)

rt_mf = rt_scf.RT_SCF(mf, 20, 200000)
rt_mf.observables.update(mag=True, energy=True, mo_occ=True, dipole=False, charge=True)

delta_field = deltaField(rt_mf, [0.0001, 0.0001, 0.0001])

rt_mf.add_potential(delta_field)

rt_mf.kernel()

