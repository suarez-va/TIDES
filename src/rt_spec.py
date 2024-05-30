import numpy as np
from numpy.fft import fft
from pyscf import lib

'''
Real-time SCF spectra
'''

c = lib.param.LIGHT_SPEED

def abs_spec(rt_mf, kick_str=1, pad=None, damp=None, preprocess_zero=True):
    '''
    Performs 1D Fourier Transform on f(t) --> f(w) for given SCF object.
    Adapted from NWChem's fft1d.m GNU Octave script (Kenneth Lopata), which can be found at https://nwchemgit.github.io/RT-TDDFT.html#absorption-spectrum-of-water
    '''

    time = []
    dipolex_t = []
    dipoley_t = []
    dipolez_t = []
    openfile = F'{rt_mf.filename}' + '_dipole.txt'

    file = open(openfile, 'r')
    lines = file.readlines()
    for line in lines:
        data = line.split('\t')
        t = data[0]
        dip = data[1][2:-2]
        x = dip.split()[0]
        y = dip.split()[1]
        z = dip.split()[2]
        time.append(t)
        dipolex_t.append(x)
        dipoley_t.append(y)
        dipolez_t.append(z)

    time = np.asarray(time).astype(np.float64)
    dipolex_t = np.asarray(dipolex_t).astype(np.float64)
    dipoley_t = np.asarray(dipoley_t).astype(np.float64)
    dipolez_t = np.asarray(dipolez_t).astype(np.float64)

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
    im_dipole_f = im_dipole_f[:m]

    osc_str = (4 * np.pi) / (3 * c * kick_str) * w * im_dipole_f

    abs_vs_freq = np.transpose([w,osc_str])
    np.savetxt(rt_mf.filename + "_abs_vs_freq.txt", abs_vs_freq)
