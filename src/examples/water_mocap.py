import numpy as np
from pyscf import gto, scf, dft
from rt_scf import RT_SCF
from rt_cap import MOCAP
from rt_vapp import ElectricField

mol = gto.M(
	verbose = 0,
	atom='''
  O     0.00000043     0.11188833     0.00000000
  H     0.76000350    -0.47275229     0.00000000
  H    -0.76000393    -0.47275063     0.00000000
  ''',
	basis='6-31G*',
    spin = 0)

mf = dft.UKS(mol)
mf.xc = 'PBE0'
mf.kernel()

rt_mf = RT_SCF(mf, 0.2, 250)
rt_mf.observables.update(dipole=True)
delta_field = ElectricField('delta', [0.0, 0.0, 0.0001])

rt_mf.prop = 'magnus_interpol'

CAP = MOCAP(1.0, 0.5, 1.0, 100.0, mf.get_ovlp())
rt_mf.add_potential(CAP, delta_field)


rt_mf.kernel()

