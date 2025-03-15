"""
Custom time-dependent potential + polarizability
"""

import numpy as np
from pyscf import gto, scf
from tides import rt_scf
from tides.rt_vapp import ElectricField

H_ATOM="H 0 0 0"
WATER='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  '''

mol = gto.M(
    verbose = 0,
    atom=WATER,
	basis='sto-3g',
    spin = 0,
)

ghf= scf.GHF(mol)
ghf.kernel()

class cosfield4C:
    """Oscillating E-field for 4C"""
    def __init__(self, rt_scf, amplitude, frequency):
        self.amplitude = np.array(amplitude)
        self.frequency = frequency
        mol = rt_scf._scf.mol
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        mol.set_common_orig_(nuc_charge_center)

        tdip = np.zeros((3, mol.nao*2, mol.nao*2))
        tdip[:,:mol.nao,:mol.nao] = -1 * mol.intor_symmetric('int1e_r', comp=3)
        tdip[:,mol.nao:,mol.nao:] = -1 * mol.intor_symmetric('int1e_r', comp=3)
        self.tdip = tdip

    def calculate_potential(self, rt_scf):
        self.field = np.cos(self.frequency * rt_scf.current_time)
        # add the linear ramp for the first period
        if (rt_scf.current_time < 2 * np.pi / self.frequency):
            self.field *= rt_scf.current_time * self.frequency / (2 * np.pi)
        return -1 * np.einsum('xij,x->ij', self.tdip, self.field*self.amplitude)

from tides import rt_output
def _custom_dipole(rt_scf):
    dipole = rt_scf._dipole
    rt_scf._log.note(f'Total Dipole Moment [X, Y, Z] (AU): {" ".join(map(str,dipole))} \n')
    strengths = np.asarray([pot.field for pot in rt_scf._potential])
    field = rt_scf._potential[0].amplitude * strengths[0]
    rt_scf._log.note(f'Field Strength [X, Y, Z] (AU): {" ".join(map(str,field))} \n')
    print("t =", rt_scf.current_time)

# hijack the dipole printer :)
rt_output._print_dipole = _custom_dipole

# Frequency-dependent polarizability
# μ¹(t) = α(-ω;ω)cos(ωt)
# μ¹_{ij} is the first-order susceptibility or dipole moment when i=j.
# polarizability α is obtained by dividing by the applied potential

fields = {
    "x_pos": [ 1e-4, 0, 0],
    "x_neg": [-1e-4, 0, 0],
    "y_pos": [ 0, 1e-4, 0],
    "y_neg": [ 0,-1e-4, 0],
    "z_pos": [ 0, 0, 1e-4],
    "z_neg": [ 0, 0,-1e-4],
}

frequency = 0.0428 # 1048nm laser
end_time  = 750

for fn, amp in fields.items():
    rt_mf = rt_scf.RT_SCF(ghf, 0.08, end_time, filename=fn)
    rt_mf.observables.update(energy=True, dipole=True)
    rt_mf.add_potential(cosfield4C(rt_mf, amp, frequency))
    rt_mf.kernel()

