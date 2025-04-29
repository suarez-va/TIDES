import numpy as np
from pyscf import gto, scf, dft
from tides import rt_scf
from tides.rt_spec import abs_spec
from tides.rt_vapp import ElectricField

mol = gto.M(
    verbose = 0,
    atom='''
 C                  0.00000000    0.00000000    0.00000000
 C                  0.00000000    0.00000000    1.39130000
 C                  1.20490114    0.00000000    2.08695000
 C                  2.40980229    0.00000000    1.39130000
 C                  2.40980229    0.00000000   -0.00000000
 C                  1.20490114    0.00000000   -0.69565000
 H                 -0.93608686   -0.00000000   -0.54045000
 H                 -0.93608686    0.00000000    1.93175000
 H                  1.20490114    0.00000000    3.16785000
 H                  3.34588915   -0.00000000    1.93175000
 H                  3.34588915   -0.00000000   -0.54045000
 H                  1.20490114   -0.00000000   -1.77655000

''',
    basis='6-31G*',
    spin = 0)

benzene = dft.RKS(mol)
benzene.xc = 'B3LYP'
benzene.kernel()

rt_benzene = rt_scf.RT_SCF(benzene, 0.2, 5000)
rt_benzene.observables.update(dipole=True)

delta_field = ElectricField('delta', [0.0000, 0.0000, 0.0001])

rt_benzene.add_potential(delta_field)

rt_benzene.kernel()
