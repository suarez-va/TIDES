import pytest
from pathlib import Path
from pyscf import gto, scf
import numpy as np
from tides import RT_Ehrenfest
from tides.analysis import parse_rt

dir_path = str(Path(__file__).resolve().parent)

FINAL_POS_REF = [[9.97211553555757e-25, -6.478232386024886e-24, -0.0024091166325111426],
                 [-9.829488659668802e-24, -3.239748945538169e-24, 1.4197037100563072]]
FINAL_VEL_REF = [[3.664099061386856e-24, -8.954169800388176e-23, -0.012045217326817922],
                 [-5.476632521966652e-23, -3.056080770049644e-23, 0.012045217326817922]]

def test_h2_ehrenfest():
    # Shortened version of TIDES/examples/H2_2c-Ehrenfest/2cB3LYP_Ehrenfest/H2_Ehrenfest.py
    # Same molecule/method/initial velocities, just far fewer propagation steps so it runs quickly.
    mol = gto.M(atom='''
      H    0.0 0.0 0.0
      H    0.0 0.0 0.75
            ''', basis='6-31G')

    gks = scf.GKS(mol)
    gks.xc = 'B3LYP'

    gdm_guess = np.array([[0.18568439, 0.28537676, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                          [0.28537676, 0.43859311, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                          [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                          [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                          [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                          [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                          [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.18568439, 0.28537676],
                          [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.28537676, 0.43859311]])
    gks.kernel(dm0=gdm_guess)

    rt_ehrenfest = RT_Ehrenfest(gks, 0.04, 0.2, Ne_step=1, N_step=1,
                                filename=dir_path + '/output.out', frequency=1,
                                chkfile=None, verbose=5)
    rt_ehrenfest.observables.update(energy=True, mag=True, nuclei=False)

    # 7.25eV of vibrational energy within the H-H bond, split evenly between the two H atoms.
    rt_ehrenfest.nuc.mass[0] = 1836.15267343
    rt_ehrenfest.nuc.mass[1] = 1836.15267343

    KE_i = 3.625
    init_velo = np.sqrt(2 * (KE_i / 27.211386246) / 1836.15267343)
    rt_ehrenfest.nuc.vel[0, 2] = -1 * init_velo
    rt_ehrenfest.nuc.vel[1, 2] = init_velo

    rt_ehrenfest.kernel()

    # parse data for comparison
    data = parse_rt.parse_output(dir_path + '/output.out')
    data_ref = parse_rt.parse_output(dir_path + '/output.ref')

    assert round(data['energy'][-1], 8) == round(data_ref['energy'][-1], 8)
    assert round(data['mag'][-1, 2], 8) == round(data_ref['mag'][-1, 2], 8)

    # confirm final nuclear positions/velocities match reference to 8 digits
    assert np.allclose(rt_ehrenfest.nuc.pos, np.array(FINAL_POS_REF), atol=1e-8)
    assert np.allclose(rt_ehrenfest.nuc.vel, np.array(FINAL_VEL_REF), atol=1e-8)
