#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some functionality useful for calculating the adjoint sources.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np


def matlab_range(start, stop, step):
    """
    Simple function emulating the behaviour of Matlab's colon notation.

    This is very similar to np.arange(), except that the endpoint is included
    if it would be the logical next sample. Useful for translating Matlab code
    to Python.
    """
    # Some tolerance
    if (abs(stop - start) / step) % 1 < 1E-7:
        return np.linspace(start, stop,
                           int(round((stop - start) / step)) + 1,
                           endpoint=True)
    return np.arange(start, stop, step)


def get_dispersed_wavetrain(dw=0.001, distance=1500.0, t_min=0, t_max=900, a=4,
                            b=1, c=1, body_wave_factor=0.01,
                            body_wave_freq_scale=0.5):
    """
    :type dw: float, optional
    :param dw: Angular frequency spacing. Defaults to 1E-3.
    :type distance: float, optional
    :param distance: The event-receiver distance in kilometer. Defaults to
        1500.
    :type t_min: float, optional
    :param t_min: The start time of the returned trace relative to the event
        origin in seconds. Defaults to 0.
    :type t_max: float, optional
    :param t_max: The end time of the returned trace relative to the event
        origin in seconds. Defaults to 900.
    :type a: float, optional
    :param a: Offset of dispersion curve. Defaults to 4.
    :type b: float, optional
    :param b: Linear factor of the dispersion curve. Defaults to 1.
    :type c: float, optional
    :param c: Quadratic factor of the dispersion curve. Defaults to 1.
    :type body_wave_factor: float, optional
    :param body_wave_factor: The factor of the body waves. Defaults to 0.01.
    :type body_wave_freq_scale: float, optional
    :param body_wave_freq_scale:  Determines the frequency of the body waves.
        Defaults to 0.5
    :returns: The time array t and the displacement array u.
    :rtype: Tuple of two numpy arrays
    """
    # Time and frequency axes
    w_min = 2.0 * np.pi / 50.0
    w_max = 2.0 * np.pi / 10.0
    w = matlab_range(w_min, w_max, dw)
    t = matlab_range(t_min, t_max, 1)

    # Define the dispersion curves.
    c = a - b * w - c * w ** 2

    # Time integration
    u = np.zeros(len(t))

    for _i in xrange(len(t)):
        u[_i] = np.sum(w * np.cos(w * t[_i] - w * distance / c) * dw)

    # Add body waves
    u += body_wave_factor * np.sin(body_wave_freq_scale * t) * \
        np.exp(-(t - 250) ** 2 / 500.0)

    return t, u


def cross_correlation(f, g):
    """
    Computes a cross correlation similar to numpy's "full" correlation, except
    shifted indices.

    :type f: numpy array
    :param f: function 1
    :type g: numpy array
    :param g: function 1
    """
    cc = np.correlate(f, g, mode="full")
    N = len(cc)
    cc_new = np.zeros(N)

    cc_new[0: (N + 1) / 2] = cc[(N + 1) / 2 - 1: N]
    cc_new[(N + 1) / 2: N] = cc[0: (N + 1) / 2 - 1]
    return cc_new


def gaussian_window(y, width):
    """
    Returns a simple gaussian window along a given axis.

    :type y: numpy array
    :param y: The values at which to compute the window.
    :param width: float
    :param width: variance = (width ^ 2) / 2
    """
    return 1.0 / (np.pi * width ** 2) ** (0.25) * \
        np.exp(-0.5 * y ** 2 / width ** 2)
