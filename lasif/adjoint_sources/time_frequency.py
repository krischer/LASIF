#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time frequency functions.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
from scipy.interpolate import interp1d

from lasif.adjoint_sources import utils


def time_frequency_transform(t, s, dt_new, width, threshold):
    """
    :param t: discrete time
    :param s: discrete signal
    :param dt_new: time increment in the tf domain
    :param width: width of the Gaussian window
    :param threshold: fraction of the absolute signal below which the Fourier
        transform is set to zero in order to reduce computation time
    """
    # New time axis
    ti = utils.matlab_range(t[0], t[-1], dt_new)
    # Interpolate both signals to the new time axis
    si = interp1d(t, s, kind=1)(ti)

    # Rename some variables
    t = ti
    s = si
    t_min = t[0]

    # Initialize the meshgrid
    N = len(t)
    dnu = 1.0 / (N * dt_new)

    nu = utils.matlab_range(0, float(N - 1) / (N * dt_new), dnu)
    tau = t
    TAU, NU = np.meshgrid(tau, nu)

    # Compute the time frequency representation
    tfs = np.zeros(NU.shape, dtype="complex128")

    for k in xrange(len(tau)):
        # Window the signals
        w = utils.gaussian_window(t - tau[k], width)
        f = w * s

        if np.abs(f).max() > (threshold * np.abs(s).max()):
            tfs[k, :] = np.fft.fft(f) / np.sqrt(2.0 * np.pi) * dt_new
            tfs[k, :] = tfs[k, :] * np.exp(-2.0 * np.pi * 1j * t_min * nu)
        else:
            tfs[k, :] = 0.0
    return TAU, NU, tfs
