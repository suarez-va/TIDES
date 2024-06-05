import numpy as np
from pyscf import gto, scf, dft
import rt_scf
from rt_spec import abs_spec
from rt_vapp import ElectricField
from rt_parse import parse

mol = gto.M(
	verbose = 0,
	atom='''
  O     0.00000000    -0.00001441    -0.34824012
  H    -0.00000000     0.76001092    -0.93285191
  H     0.00000000    -0.75999650    -0.93290797
  ''',
	basis='6-31G',
    spin = 0)

mf = dft.RKS(mol)
mf.xc = 'PBE0'
mf.kernel()

rt_mf = rt_scf.RT_SCF(mf, 0.2, 1, 1000, 'water_abs')
rt_mf.observables.update(dipole=True)

delta_field = ElectricField('delta', [0.0001, 0.0001, 0.0001])

rt_mf.add_potential(delta_field)

rt_mf.kernel()

parse(rt_mf)
time = rt_mf.dipole[:,0]
dipole_xyz = rt_mf.dipole[:,1:]
abs_spec(time, dipole_xyz, 'water_abs_values', 0.0001, 50000, 50) # Zero-pad and exponential damping for clean spectrum
