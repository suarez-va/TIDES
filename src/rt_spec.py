import numpy as np
from numpy.fft import fft
from pyscf import lib


'''
Real-time SCF Spectra
'''

c = lib.param.LIGHT_SPEED

def abs_spec(time, dipole, filename, kick_str=1, pad=None, damp=None, preprocess_zero=True):
    '''
    Performs 1D Fourier Transform on time-dependent dipole moment.
    Adapted from NWChem's fft1d.m GNU Octave script (Kenneth Lopata), which can be found at https://nwchemgit.github.io/RT-TDDFT.html#absorption-spectrum-of-water
    '''

    dipolex_t = np.copy(dipole[:,0])
    dipoley_t = np.copy(dipole[:,1])
    dipolez_t = np.copy(dipole[:,2])

    if preprocess_zero:
        dipolex_t -= dipolex_t[0]
        dipoley_t -= dipoley_t[0]
        dipolez_t -= dipolez_t[0]

    if damp:
        d = np.exp((-1 * (time - time[0])) / damp)
        dipolex_t *= d
        dipoley_t *= d
        dipolez_t *= d

    if pad:
        zeros = np.linspace(0,0, pad)
        dipolex_t = np.append(dipolex_t,zeros)
        dipoley_t = np.append(dipoley_t,zeros)
        dipolez_t = np.append(dipolez_t,zeros)

    n = len(dipolex_t)
    dt = time[1] - time[0]              # Assumes constant timestep
    period = (n-1) * dt - time[0]
    dw = 2.0 * np.pi / period

    m = int(n / 2)                      # Include only positive frequencies
    wmin = 0.0
    wmax = m * dw
    w = np.linspace(wmin, wmax, m)
    dipolex_f = fft(dipolex_t)
    dipoley_f = fft(dipoley_t)
    dipolez_f = fft(dipolez_t)

    im_dipole_f = np.imag(dipolex_f) + np.imag(dipoley_f) + np.imag(dipolez_f)
    
    im_dipole_f = np.abs(im_dipole_f[:m])
    osc_str = (4 * np.pi) / (3 * c * kick_str) * w * im_dipole_f
    abs_vs_freq = np.transpose([w,osc_str])
    np.savetxt(filename + '.txt', abs_vs_freq)
