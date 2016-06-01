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
from obspy.signal.interpolation import lanczos_interpolation
import scipy.fftpack

from lasif.adjoint_sources import utils



def time_frequency_transform(t, s, dt_new, width, threshold=1E-5):
    """
    Gabor transform (time frequency transform with Gaussian windows).

    Data will be resampled before it is transformed.

    :param t: discrete time
    :param s: discrete signal
    :param dt_new: Signal will be resampled to that time interval.
    :param width: width of the Gaussian window
    :param threshold: fraction of the absolute signal below which the Fourier
        transform is set to zero in order to reduce computation time
    """
    # New time axis
    ti = utils.matlab_range(t[0], t[-1], dt_new)
    # Interpolate both signals to the new time axis
    si = lanczos_interpolation(
        data=s, old_start=t[0], old_dt=t[1] - t[0], new_start=t[0],
        new_dt=dt_new, new_npts=len(ti), a=8, window="blackmann")

    # Rename some variables
    t = ti
    s = si
    assert t[0] == 0
    # If we ever get signals not starting at time 0: uncomment this line.
    # t_min = t[0]

    # Initialize the meshgrid
    N = len(t)

    nu = np.linspace(0, float(N - 1) / (N * dt_new), N)

    # Compute the time frequency representation
    tfs = np.zeros((N, N), dtype="complex128")

    threshold = np.abs(s).max() * threshold

    for k in xrange(N):
        # Window the signals
        f = utils.gaussian_window(t - t[k], width) * s

        # No need to transform if nothing is there. Great speedup as lots of
        # windowed functions have 0 everywhere.
        # if np.abs(f).max() < threshold:
        #     continue

        tfs[k, :] = scipy.fftpack.fft(f)
        # If we ever get signals not starting at time 0: uncomment this line.
        # tfs[k, :] = tfs[k, :] * np.exp(-2.0 * np.pi * 1j * t_min * nu)

    tfs *= dt_new / np.sqrt(2.0 * np.pi)

    return t, nu, tfs


def time_frequency_cc_difference(t, s1, s2, dt_new, width):
    """
    Straight port of tfa_cc_new.m

    :param t: discrete time
    :param s1: discrete signal 1
    :param s2: discrete signal 2
    :param dt_new: time increment in the tf domain
    :param width: width of the Gaussian window
    :param threshold: fraction of the absolute signal below which the Fourier
        transform is set to zero in order to reduce computation time
    """
    # New time axis
    ti = utils.matlab_range(t[0], t[-1], dt_new)
    # Interpolate both signals to the new time axis
    si1 = lanczos_interpolation(
        data=s1, old_start=t[0], old_dt=t[1] - t[0], new_start=t[0],
        new_dt=dt_new, new_npts=len(ti), a=8, window="blackmann")
    si2 = lanczos_interpolation(
        data=s2, old_start=t[0], old_dt=t[1] - t[0], new_start=t[0],
        new_dt=dt_new, new_npts=len(ti), a=8, window="blackmann")

    # Extend the time axis, required for the correlation
    N = len(ti)
    t_cc = np.linspace(t[0], t[0] + (2 * N - 2) * dt_new, (2 * N - 1))

    # Rename some variables
    s1 = si1
    s2 = si2

    assert t_cc[0] == 0
    # If we ever get signals not starting at time 0: uncomment this line.
    # t_min = t_cc[0]

    N = len(t_cc)
    dnu = 1.0 / (N * dt_new)

    nu = np.linspace(0, (N - 1) * dnu, N)
    tau = t_cc

    # Compute the time frequency representation
    tfs = np.zeros((nu.shape[0], tau.shape[0]), dtype="complex128")

    for k in xrange(len(tau)):
        # Window the signals
        w = utils.gaussian_window(ti - tau[k], width)
        f1 = w * s1
        f2 = w * s2

        cc = utils.cross_correlation(f2, f1)
        tfs[k, :] = scipy.fftpack.fft(cc)
        # If we ever get signals not starting at time 0: uncomment this line.
        # tfs[k, :] = tfs[k, :] * np.exp(-2.0 * np.pi * 1j * t_min * nu)

    tfs *= dt_new / np.sqrt(2.0 * np.pi)

    return tau, nu, tfs


def itfa(tau, tfs, width):
    # initialisation
    N = len(tau)
    dt = tau[1] - tau[0]

    assert tau[0] == 0

    # If we ever get signals not starting at time 0: uncomment these lines.
    # t_min = tau[0]
    # for k in xrange(len(tau)):
    #     tfs[k, :] = tfs[k, :] * np.exp(2.0 * np.pi * 1j * nu.transpose() *
    #                                    t_min)

    # inverse fft
    I = np.zeros((N, N), dtype="complex128")

    for k in xrange(N):
        I[k, :] = scipy.fftpack.ifft(tfs[k, :])

    I *= 2.0 * np.pi / dt

    # time integration
    s = np.zeros(N, dtype="complex128")

    for k in xrange(N):
        f = utils.gaussian_window(tau[k] - tau, width) * I[:, k].transpose()
        s[k] = np.sum(f) * dt

    s /= np.sqrt(2.0 * np.pi)

    return s, tau, I
