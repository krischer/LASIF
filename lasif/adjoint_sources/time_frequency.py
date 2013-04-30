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


def time_frequency_cc_difference(t, s1, s2, dt_new, width, threshold):
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
    si1 = interp1d(t, s1, kind=1)(ti)
    si2 = interp1d(t, s2, kind=1)(ti)

    # Extend the time axis, required for the correlation
    N = len(ti)
    t_cc = utils.matlab_range(t[0], (2 * N - 2) * dt_new, dt_new)

    # Rename some variables
    s1 = si1
    s2 = si2
    t_min = t_cc[0]

    # Initialize the meshgrid
    N = len(t_cc)
    dnu = 1.0 / (N * dt_new)

    nu = utils.smatlab_range(0, float(N - 1) / (N * dt_new), dnu)
    tau = t_cc
    TAU, NU = np.meshgrid(tau, nu)

    # Compute the time frequency representation
    tfs = np.zeros(NU.shape, dtype="complex128")

    for k in xrange(len(tau)):
        # Window the signals
        w = utils.gaussian_window(ti - tau[k], width)
        f1 = w * s1
        f2 = w * s2

        if np.abs(f1).max() > (threshold * np.abs(s1).max()):
            cc = utils.cross_correlation(f2, f1)
            tfs[k, :] = np.fft.fft(cc) / np.sqrt(2.0 * np.pi) * dt_new
            tfs[k, :] = tfs[k, :] * np.exp(-2.0 * np.pi * 1j * t_min * nu)
        else:
            tfs[k, :] = 0.0
    return TAU, NU, tfs


def itfa(TAU, NU, tfs, width, threshold):

    #%========================================================================
    #% initialisation
    #%========================================================================

    tau = TAU[0, :]
    nu = NU[:, 0]

    N = len(tau)
    dt = tau[1] - tau[0]

    #%========================================================================
    #% modification of the signal
    #%========================================================================

    #% Zeitachse: tfs(k,:)
    #% Frequenzachse: tfs(:,1)

    t_min = tau[0]

    for k in xrange(len(tau)):
        tfs[k, :] = tfs[k, :] * np.exp(2.0 * np.pi * 1j * nu.transpose() *
                t_min)

    #%========================================================================
    #% inverse fft
    #%========================================================================
    I = np.zeros((N, N), dtype="complex128")

    max_tfs = np.abs(tfs).max()

    for k in xrange(N):
        if np.abs(tfs[k, :]).max() > threshold * max_tfs:
            I[k, :] = 2.0 * np.pi * np.fft.ifft(tfs[k, :]) / dt
        else:
            I[k, :] = 0.0

    #%========================================================================
    #% time integration
    #%========================================================================

    s = np.zeros(N, dtype="complex128")

    for k in xrange(N):
        f = utils.gaussian_window(tau[k] - tau, width) * I[:, k].transpose()
        s[k] = np.sum(f) * dt

    s /= np.sqrt(2.0 * np.pi)

    return s, tau, I
