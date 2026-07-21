import pytest
import numpy as np
from pathlib import Path
from pyscf import gto, dft
from tides import RT_SCF
from tides import ElectricField
from tides.analysis import parse_rt

dir_path = str(Path(__file__).resolve().parent)

def test_water_uvvis():
    # Shortened version of TIDES/examples/Water_RKS_UV-Vis/Water_RKS_UV-Vis.py
    # Same molecule/method/delta-field kick, just far fewer propagation steps so it runs quickly.
    mol = gto.M(
        verbose = 0,
        atom='''
      O     0.00000000    -0.00001441    -0.34824012
      H    -0.00000000     0.76001092    -0.93285191
      H     0.00000000    -0.75999650    -0.93290797
      ''',
        basis='6-31G',
        spin = 0)

    rks = dft.RKS(mol)
    rks.xc = 'PBE0'
    rks.kernel()

    rt_scf = RT_SCF(rks, 0.2, 4, filename=dir_path + '/output.out', chkfile=None)
    rt_scf.observables.update(dipole=True)

    delta_field = ElectricField('delta', [0.0001, 0.0001, 0.0001]) # Applying x,y,z polarization simultaneously
    rt_scf.add_potential(delta_field)

    rt_scf.kernel()

    # parse data for comparison
    data = parse_rt.parse_output(dir_path + '/output.out')
    data_ref = parse_rt.parse_output(dir_path + '/output.ref')

    # confirm final dipole moment equal to 8 digits
    assert np.allclose(data['dipole'][-1], data_ref['dipole'][-1], atol=1e-8)
