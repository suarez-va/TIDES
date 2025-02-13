import numpy as np
from pyscf import gto, scf, dft
from rt_scf import RT_SCF
from staticfield import static_bfield
from rt_utils import input_fragments

'''
Lithium Atom in a static B-Field
Recreated from https://doi.org/10.1063/1.4902884
'''

# Here we tilt the BField
mag = -0.000085 # in au

mag_x = mag * np.sin(np.pi/4) * np.cos(np.pi/4)
mag_y = mag * np.sin(np.pi/4) * np.sin(np.pi/4)
mag_z = mag * np.cos(np.pi/4) 

# Build mol
LiMol = gto.M(
	verbose = 0,
	atom='''
 Li                 0.00000000    0.00000000 0.00000000
''',
	basis='3-21G',
    spin = 1)

# Build and run GHF object
Li = scf.ghf.GHF(LiMol)
Li.kernel()

# Create BField object
static_bfield(Li, [mag_x,mag_y,mag_z])

# Create RT_SCF object
rt_mf = RT_SCF(Li, 1, 10000)

# Specify propagator
rt_mf.prop = 'magnus_step'
# In virtually all cases, the default 'magnus_interpol' is a more robust integrator.
# However the magnus_step (or MMUT) integrator is what was used in the original paper

# Declare observables, in our case we only care about the magnetization
rt_mf.observables.update(mag=True)

# Start propagation
rt_mf.kernel()

