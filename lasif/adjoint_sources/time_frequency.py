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
import scipy.fftpack
import scipy.interpolate

from lasif.adjoint_sources import utils


def time_frequency_transform(t, s, width, threshold=1E-2):
    """
    Gabor transform (time frequency transform with Gaussian windows).

    Data will be resampled before it is transformed.

    :param t: discrete time.
    :param s: discrete signal.
    :param width: width of the Gaussian window
    :param threshold: fraction of the absolute signal below which the Fourier
        transform is set to zero in order to reduce computation time
    """
    N = len(t)
    dt = t[1] - t[0]

    nu = np.linspace(0, float(N - 1) / (N * dt), N)

    # Compute the time frequency representation
    tfs = np.zeros((N, N), dtype="complex128")

    threshold = np.abs(s).max() * threshold

    for k in range(N):
        # Window the signals
        f = utils.gaussian_window(t - t[k], width) * s

        # No need to transform if nothing is there. Great speedup as lots of
        # windowed functions have 0 everywhere.
        if np.abs(f).max() < threshold:
            continue

        tfs[k, :] = scipy.fftpack.fft(f)

    tfs *= dt / np.sqrt(2.0 * np.pi)

    return t, nu, tfs


def time_frequency_cc_difference(t, s1, s2, width, threshold=1E-2):
    """
    Straight port of tfa_cc_new.m

    :param t: discrete time
    :param s1: discrete signal 1
    :param s2: discrete signal 2
    :param width: width of the Gaussian window
    :param threshold: fraction of the absolute signal below which the Fourier
        transform is set to zero in order to reduce computation time
    """
    dt = t[1] - t[0]

    # Extend the time axis, required for the correlation
    N = len(t)
    t_cc = np.linspace(t[0], t[0] + (2 * N - 2) * dt, (2 * N - 1))

    N = len(t_cc)
    dnu = 1.0 / (N * dt)

    nu = np.linspace(0, (N - 1) * dnu, N)
    tau = t_cc

    cc_freqs = scipy.fftpack.fftfreq(len(t_cc), d=dt)
    freqs = scipy.fftpack.fftfreq(len(t), d=dt)

    # Compute the time frequency representation
    tfs = np.zeros((len(t), len(t)), dtype="complex128")

    threshold = np.abs(s1).max() * threshold

    for k in range(len(t)):
        # Window the signals
        w = utils.gaussian_window(t - tau[k], width)
        f1 = w * s1
        f2 = w * s2

        if min(np.abs(f1).max(), np.abs(f2).max()) < threshold:
            continue

        cc = utils.cross_correlation(f2, f1)
        tfs[k, :] = \
            scipy.interpolate.interp1d(cc_freqs, scipy.fftpack.fft(cc))(freqs)
    tfs *= dt / np.sqrt(2.0 * np.pi)

    return tau, nu, tfs


def itfa(tau, tfs, width, threshold=1E-2):
    N = len(tau)
    dt = tau[1] - tau[0]

    threshold = np.abs(tfs).max() * threshold

    # inverse fft
    I = np.zeros((N, N), dtype="complex128")

    # IFFT and scaling.
    for k in range(N):
        if np.abs(tfs[k, :]).max() < threshold:
            continue
        I[k, :] = scipy.fftpack.ifft(tfs[k, :])
    I *= 2.0 * np.pi / dt

    # time integration
    s = np.zeros(N, dtype="complex128")

    for k in range(N):
        f = utils.gaussian_window(tau[k] - tau, width) * I[:, k].transpose()
        s[k] = np.sum(f) * dt
    s *= dt / np.sqrt(2.0 * np.pi)

    return s, tau, I
