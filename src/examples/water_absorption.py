import numpy as np
from pyscf import gto, scf, dft
import rt_scf
import rt_spec
from rt_vapp import electric_field

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

rt_mf = rt_scf.rt_scf(mf, 0.2, 1, 1000, 'water_abs')

delta_field = electric_field('delta', [0.0001, 0.0001, 0.0001])

rt_mf.add_field(delta_field)
rt_mf.prop = 'magnus_interpol'

rt_mf.kernel()

rt_spec.abs_spec(rt_mf,0.0001,50000,50) # Zero-pad and exponential damping for clean spectrum
