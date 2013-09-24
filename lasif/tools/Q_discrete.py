#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computation and visualisation of a discrete absorption-band model.

C_r=1 and rho=1 are assumed
tau is computed from the target Q via 1/Q=0.5*tau*pi

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch),
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012-2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import matplotlib.pyplot as plt
import numpy as np
import random


def calculate_Q_model(Q, N, f_min, f_max, max_iterations=10000,
                      initial_temperature=0.1, cooling_factor=0.9998):
    """
    :type Q: float
    :param Q: The target Q value.
    :type N: int
    :param N: The number of desired relaxation mechanisms.
    :type f_min: float
    :param f_min: Minimum frequency for the discrete-case optimization in Hz.
    :type f_max: float
    :param f_max: Maximum frequency for the discrete-case optimization in Hz.
    """
    # Make logarithmic frequency axis
    f = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    w = 2.0 * np.pi * f

    # compute tau from target Q
    tau = 2.0 / (np.pi * Q)

    # computations for discrete absorption-band model

    # compute initial relaxation times: logarithmically distributed relaxation
    # times
    tau_min = 1.0 / f_max
    tau_max = 1.0 / f_min
    tau_p = np.logspace(np.log10(tau_min), np.log10(tau_max), N) / \
        (2.0 * np.pi)

    # compute initial weights
    D_p = 1.0 / tau_p
    D_p = D_p / sum(D_p)

    # compute initial sum that we later force to pi/2 over the frequency range
    # of interest
    S = 0.0

    for n in np.arange(N):
        S = S+D_p[n] * w * tau_p[n] / (1 + w ** 2 * tau_p[n] ** 2)

    # compute initial misfit
    chi = sum((S - np.pi / 2.0) ** 2)

    # random search for optimal parameters
    tau_p_test = np.array(np.arange(N), dtype=float)
    D_p_test = np.array(np.arange(N), dtype=float)

    for it in np.arange(max_iterations):

        # compute perturbed parameters
        for k in np.arange(N):
            tau_p_test[k] = tau_p[k] * (1.0 + (0.5 - random.random()) *
                                        initial_temperature)
            D_p_test[k] = D_p[k] * (1.0 + (0.5 - random.random()) *
                                    initial_temperature)

        # compute new S
        S_test = 0.0
        for k in np.arange(N):
            S_test = S_test + D_p_test[k] * w * tau_p_test[k] / \
                (1 + w ** 2 * tau_p_test[k] ** 2)

        # compute new misfit and new temperature
        chi_test = sum((S_test-np.pi / 2.0) ** 2)
        initial_temperature = initial_temperature * cooling_factor

        # check if the tested parameters are better
        if chi_test < chi:
            D_p[:] = D_p_test[:]
            tau_p[:] = tau_p_test[:]
            chi = chi_test

    # compute optimal Q model
    A = 0.0
    B = 0.0

    for n in np.arange(N):
        A = A + D_p[n] * w ** 2 * tau_p[n] ** 2 / \
            (1 + w ** 2 * tau_p[n] ** 2)
        B = B + D_p[n] * w * tau_p[n] / (1 + w ** 2 * tau_p[n] ** 2)

    A = 1 + tau*A
    B = tau * B

    Q_discrete = A / B
    v_discrete = np.sqrt(2 * (A ** 2 + B ** 2) /
                         (A + np.sqrt(A ** 2 + B ** 2)))

    return D_p, tau_p, Q_discrete, v_discrete


def plot(Q, Q_discrete, v_discrete, f_min, f_max, f_min_plot=1.0/1000.0,
         f_max_plot=1.0/1.0):
    """
    :type f_min_plot: float
    :param f_min_plot: Minimum plotted frequency in Hz.
    :type f_max_plot: float
    :param f_max_plot: Maximum plotted frequency in Hz.
    """
    f_plot = np.logspace(np.log10(f_min_plot), np.log10(f_max_plot), 100)

    # minimum frequency [Hz] for plotting
    f_min_plot = 1.0/1000.0
    # maximum frequency in [Hz] for plotting
    f_max_plot = 1.0/1.0

    # plot Q and phase velocity as function of frequency
    plt.subplot(121)
    plt.semilogx([f_min, f_min], [0.9/Q, 1.1/Q], 'r')
    plt.semilogx([f_max, f_max], [0.9/Q, 1.1/Q], 'r')
    plt.semilogx([f_min_plot, f_max_plot], [1.0/Q, 1.0/Q], 'b')
    plt.semilogx(f_plot, 1/Q_discrete, 'k', linewidth=2)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('1/Q')
    plt.title('absorption (1/Q)')

    plt.subplot(122)
    plt.semilogx([f_min, f_min], [0.9, 1.1], 'r')
    plt.semilogx([f_max, f_max], [0.9, 1.1], 'r')
    plt.semilogx(f_plot, v_discrete, 'k', linewidth=2)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('v')
    plt.title('phase velocity')

    plt.show()
