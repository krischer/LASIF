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


def calculate_Q_model(Q, N, f_min, f_max, iterations=10000,
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
    :type iterations: int, optional
    :param iterations: The number of iterations performed.
    :type initial_temperature: float, optional
    :param initial_temperature: The initial temperature for the simulated
        annealing process.
    :type cooling_factor: float, optional
    :param cooling_factor: The cooling factor for the simulated annealing.
    """
    # Make a sparse logarithmic frequency axis
    f = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    # Angular frequency.
    w = 2.0 * np.pi * f

    # computations for discrete absorption-band model
    # compute initial relaxation times: logarithmically distributed relaxation
    # times
    tau_p = np.logspace(np.log10(1.0 / f_max), np.log10(1.0 / f_min), N) / \
        (2.0 * np.pi)

    # compute initial normalized weights
    D_p = 1.0 / tau_p
    D_p /= D_p.sum()

    # compute initial sum that we later force to pi/2 over the frequency range
    # of interest
    S = 0.0
    for n in xrange(N):
        S += D_p[n] * w * tau_p[n] / (1 + w ** 2 * tau_p[n] ** 2)
    # compute initial misfit
    chi = sum((S - np.pi / 2.0) ** 2)

    # random search for optimal parameters
    tau_p_test = np.empty(N, dtype=float)
    D_p_test = np.empty(N, dtype=float)

    for _i in xrange(iterations):
        # compute perturbed parameters
        tau_p_test[:] = tau_p * (1.0 + (0.5 - np.random.random()) *
                                 initial_temperature)
        D_p_test[:] = D_p * (1.0 + (0.5 - np.random.random()) *
                             initial_temperature)

        # compute new S
        S_test = 0.0
        for k in xrange(N):
            S_test += D_p_test[k] * w * tau_p_test[k] / \
                (1 + w ** 2 * tau_p_test[k] ** 2)

        # compute new misfit and new temperature
        chi_test = ((S_test - np.pi / 2.0) ** 2).sum()
        initial_temperature *= cooling_factor

        # Check if the tested parameters are better
        if chi_test < chi:
            # Swap references. Faster then copying the values.
            D_p, D_p_test = D_p_test, D_p
            tau_p, tau_p_test = tau_p_test, tau_p
            chi = chi_test

    return D_p, tau_p


def plot(D_p, tau_p, f_min, f_max, target_Q):
    """
    :type D_p: np.ndarray
    :param D_p: The calculated D_p.
    :type tau_p: np.ndarray
    :param tau_p: The calculated tau_p.
    :type f_min: float
    :param f_min: Minimum frequency for the discrete-case optimization in Hz.
        Needed to plot the boundaries.
    :type f_max: float
    :param f_max: Maximum frequency for the discrete-case optimization in Hz.
        Needed to plot the boundaries.
    :type target_Q: float
    :param target_Q: The target Q used for the optimization.
    """
    # Calculate the plotting boundaries.
    f_min_plot = 10 ** (np.log10(f_min) - 1)
    f_max_plot = 10 ** (np.log10(f_max) + 1)
    f_plot = np.logspace(np.log10(f_min_plot), np.log10(f_max_plot), 100)

    # compute optimal Q model
    A = 0.0
    B = 0.0
    N = len(D_p)
    # Make a sparse logarithmic frequency axis
    f = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    # Angular frequency.
    w = 2.0 * np.pi * f
    tau = 2.0 / (np.pi * target_Q)

    for n in xrange(N):
        A = A + D_p[n] * w ** 2 * tau_p[n] ** 2 / \
            (1 + w ** 2 * tau_p[n] ** 2)
        B = B + D_p[n] * w * tau_p[n] / (1 + w ** 2 * tau_p[n] ** 2)

    A = 1 + tau * A
    B = tau * B

    Q_discrete = A / B
    v_discrete = np.sqrt(2 * (A ** 2 + B ** 2) /
                         (A + np.sqrt(A ** 2 + B ** 2)))

    # plot Q and phase velocity as function of frequency
    plt.subplot(121)
    plt.semilogx([f_min, f_min], [0.9 / target_Q, 1.1 / target_Q], "r")
    plt.semilogx([f_max, f_max], [0.9 / target_Q, 1.1 / target_Q], "r")
    plt.semilogx([f_min_plot, f_max_plot], [1.0 / target_Q, 1.0 / target_Q],
                 "b")
    plt.semilogx(f_plot, 1.0 / Q_discrete, "k", linewidth=2)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("1/Q")
    plt.title("absorption (1/Q)")

    plt.subplot(122)
    plt.semilogx([f_min, f_min], [0.9, 1.1], "r")
    plt.semilogx([f_max, f_max], [0.9, 1.1], "r")
    plt.semilogx(f_plot, v_discrete, "k", linewidth=2)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("v")
    plt.title("phase velocity")

    plt.show()
