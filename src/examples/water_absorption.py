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
	basis='6-31G',
    spin = 0)

mf = dft.RKS(mol)
mf.xc = 'PBE0'
mf.kernel()

rt_mf = rt_scf.RT_SCF(mf, 0.2, 1, 1000)
rt_mf.observables.update(dipole=True)

delta_field = ElectricField('delta', [0.0001, 0.0001, 0.0001])

rt_mf.add_potential(delta_field)

rt_mf.kernel()

time = rt_mf.time
dipole_xyz = rt_mf.dipole
xdip = np.vstack((rt_mf.time, rt_mf.dipole[:, 0])).T
ydip = np.vstack((rt_mf.time, rt_mf.dipole[:, 1])).T
zdip = np.vstack((rt_mf.time, rt_mf.dipole[:, 2])).T
np.savetxt("xdip.txt", xdip)
np.savetxt("ydip.txt", ydip)
np.savetxt("zdip.txt", zdip)
abs_spec(time, dipole_xyz, 'water_abs_values', 0.0001, 50000, 50) # Zero-pad and exponential damping for clean spectrum
