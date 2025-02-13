import numpy as np
from pyscf import gto, scf, dft
import cProfile
import pstats
from rt_scf import RT_SCF
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

rt_mf = RT_SCF(mf, 0.2, 1000)
rt_mf.observables.update(energy=True, dipole=True)

gaussian_field = ElectricField('gaussian', amplitude=[0.0, 0.0, 0.0001],
                             center=393.3, frequency=0.3768,
                              width=64.8)

rt_mf.add_potential(gaussian_field)
rt_mf.kernel()
