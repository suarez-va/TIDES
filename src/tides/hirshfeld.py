import sys
import numpy as np
from pyscf import gto, scf, dft, data
from pyscf.scf.atom_ks import get_atm_nrks
from pyscf.dft import numint

try:
    # pip install git+https://github.com/frobnitzem/hirshfeld
    from pyscf.hirshfeld import HirshfeldAnalysis
except ImportError:
    sys.stderr.write('Note: Hirshfeld module not installed. Install with [pip install git+https://github.com/frobnitzem/hirshfeld] if you wish to collect hirshfeld observables.')

def hirshfeld_partition(mf, den_ao, grids=None, atom_weights=None):
    if grids is None or atom_weights is None: grids, atom_weights = get_weights(mf.mol)
    den_ao = den_ao.astype(np.complex128)
    
    # Restricted
    if mf.istype('RHF') | mf.istype('RKS'):
        rho = cast_den_ao(mf, den_ao, grids)
        return rho * atom_weights
    # Unrestricted - break into aa, bb spin blocks
    if mf.istype('UHF') | mf.istype('UKS'):
        rho_a = cast_den_ao(mf, den_ao[0], grids)
        rho_b = cast_den_ao(mf, den_ao[1], grids)
        return rho_a * atom_weights, rho_b * atom_weights
    # Generalized - break into aa, ab, ba, bb spin blocks
    if mf.istype('GHF') | mf.istype('GKS'):
        ovlp = mf.get_ovlp()
        Nsp = int(ovlp.shape[0]/2)

        rho_aa = cast_den_ao(mf, den_ao[:Nsp, :Nsp], grids)
        rho_ab = cast_den_ao(mf, den_ao[:Nsp, Nsp:], grids)
        rho_ba = cast_den_ao(mf, den_ao[Nsp:, :Nsp], grids)
        rho_bb = cast_den_ao(mf, den_ao[Nsp:, Nsp:], grids)
        return rho_aa * atom_weights, rho_ab * atom_weights, rho_ba * atom_weights, rho_bb * atom_weights

def get_mag(rho_aa, rho_ab, rho_ba, rho_bb):
    mx = (rho_ab + rho_ba)
    my = 1j * (rho_ab - rho_ba)
    mz = (rho_aa - rho_bb)
    return mx, my, mz

def cast_den_ao(mf, den_ao, grids):
    # Re Part
    rho_re = numint.get_rho(numint.NumInt(), mf.mol, np.real(den_ao), grids)
    # Imag Part
    rho_imag = numint.get_rho(numint.NumInt(), mf.mol, np.imag(den_ao), grids)

    return (rho_re + 1j * rho_imag) * grids.weights

def get_weights(mol):
    grids = dft.Grids(mol)
    grids.build()
    mf = dft.RKS(mol)
    mf.verbose = 0
    mf.xc = 'HF'
    mf.grids = grids
    mf.kernel()

    H = HirshfeldAnalysis(mf).run()
    return grids, H.result['weights_free']
