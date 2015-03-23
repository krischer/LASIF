#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computation and visualisation of a discrete absorption-band model.

For a given array of target Q values, the code determines the optimal
relaxation times and weights. This is done within in specified frequency range.

To the large this is from the tools shipping with SES3D.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch),
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012-2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd


def calculate_Q_model(N, f_min, f_max, iterations=30000,
                      initial_temperature=0.2, cooling_factor=0.9998,
                      quiet=False):
    """
    :type N: int
    :param N: The number of desired relaxation mechanisms. The broader the
        absorption band, the more mechanisms are needed.
    :type f_min: float
    :param f_min: Minimum frequency for the discrete-case optimization in
        Hz. Lower frequency limit of the absorption band.
    :type f_max: float
    :param f_max: Maximum frequency for the discrete-case optimization in
        Hz. Upper frequency limit of the absorption band.
    :type iterations: int
    :param iterations: The number of iterations performed.
    :type initial_temperature: float
    :param initial_temperature: The initial temperature for the simulated
        annealing process.
    :type cooling_factor: float
    :param cooling_factor: The cooling factor for the simulated annealing.
    :type quiet: bool
    :param quiet: Whether or not to be quiet.
    """
    # Array of target Q's at the reference frequency (f_ref, specified below).
    # The code tries to find optimal relaxation parameters for all given Q_0
    # values simultaneously.
    Q_0 = np.array([50.0, 100.0, 500.0])

    # Number of relaxation mechanisms. The broader the absorption band,
    # the more mechanisms are needed.
    N = 3

    # Optimisation parameters (number of iterations, temperature,
    # temperature decrease). The code runs a simplistic Simulated Annealing
    # optimisation to find optimal relaxation parameters. max_it is the
    # maximum number of samples, T_0 the initial random step length,
    # and d is the temperature decrease in the sense that temperature
    # decreases from one sample to the next by a factor of d.
    max_it = iterations
    T_0 = initial_temperature
    d = cooling_factor

    # Reference frequency in Hz (f_ref) and exponent (alpha) for
    # frequency-dependent Q. For frequency-independent Q you must set
    # alpha=0.0.
    f_ref = f_max - (f_max - f_min) * 0.1
    alpha = 0.0

    # initialisations

    # make logarithmic frequency axis
    f = np.logspace(np.log10(f_min), np.log10(f_max), 100)
    w = 2.0 * np.pi * f

    # compute tau from target Q at reference frequency
    tau = 1.0 / Q_0

    # compute target Q as a function of frequency
    Q_target = np.zeros([len(Q_0), len(f)])

    for n in range(len(Q_0)):
        Q_target[n, :] = Q_0[n] * (f / f_ref) ** alpha

    # compute initial relaxation times: logarithmically distributed
    tau_min = 1.0 / f_max
    tau_max = 1.0 / f_min
    tau_s = np.logspace(np.log10(tau_min), np.log10(tau_max), N) / (2 * np.pi)

    # make initial weights
    D = np.ones(N)

    if not quiet:
        print "Starting to find optimal relaxation parameters."

    # STAGE I
    # Compute relaxation times for constant Q values and weights all equal

    # compute initial Q
    chi = 0.0

    for n in np.arange(len(Q_0)):
        A = 1.0
        B = 0.0
        for p in np.arange(N):
            A += tau[n] * (D[p] * w ** 2 * tau_s[p] ** 2) / (
                1.0 + w ** 2 * tau_s[p] ** 2)
            B += tau[n] * (D[p] * w * tau_s[p]) / (
                1.0 + w ** 2 * tau_s[p] ** 2)

        Q = A / B
        chi += sum((Q - Q_0[n]) ** 2 / Q_0[n] ** 2)

    # search for optimal parameters
    T = T_0

    for _ in xrange(max_it):
        # compute perturbed parameters
        tau_s_test = tau_s * (1.0 + (0.5 - rd.rand(N)) * T)
        D_test = D * (1.0 + (0.5 - rd.rand(1)) * T)

        # compute test Q
        chi_test = 0.0

        for n in np.arange(len(Q_0)):
            A = 1.0
            B = 0.0
            for p in np.arange(N):
                A += tau[n] * (D_test[p] * w ** 2 * tau_s_test[p] ** 2) / (
                    1.0 + w ** 2 * tau_s_test[p] ** 2)
                B += tau[n] * (D_test[p] * w * tau_s_test[p]) / (
                    1.0 + w ** 2 * tau_s_test[p] ** 2)

            Q_test = A / B
            chi_test += sum((Q_test - Q_0[n]) ** 2 / Q_0[n] ** 2)

        # compute new temperature
        T = T * d

        # check if the tested parameters are better
        if chi_test < chi:
            D[:] = D_test[:]
            tau_s[:] = tau_s_test[:]
            chi = chi_test

    # STAGE II
    # Compute weights for frequency-dependent Q with relaxation times fixed

    # compute initial Q
    chi = 0.0

    for n in np.arange(len(Q_0)):
        A = 1.0
        B = 0.0
        for p in np.arange(N):
            A += tau[n] * (D[p] * w ** 2 * tau_s[p] ** 2) / (
                1.0 + w ** 2 * tau_s[p] ** 2)
            B += tau[n] * (D[p] * w * tau_s[p]) / (
                1.0 + w ** 2 * tau_s[p] ** 2)
        Q = A / B
        chi += sum((Q - Q_target[n, :]) ** 2 / Q_0[n] ** 2)

    # random search for optimal parameters

    T = T_0

    for _ in xrange(max_it):
        # compute perturbed parameters
        D_test = D * (1.0 + (0.5 - rd.rand(N)) * T)
        # compute test Q
        chi_test = 0.0

        for n in np.arange(len(Q_0)):
            A = 1.0
            B = 0.0
            for p in np.arange(N):
                A += tau[n] * (D_test[p] * w ** 2 * tau_s[p] ** 2) / (
                    1.0 + w ** 2 * tau_s[p] ** 2)
                B += tau[n] * (D_test[p] * w * tau_s[p]) / (
                    1.0 + w ** 2 * tau_s[p] ** 2)

            Q_test = A / B
            chi_test += sum((Q_test - Q_target[n, :]) ** 2 / Q_0[n] ** 2)

        # compute new temperature
        T = T * d

        # check if the tested parameters are better
        if chi_test < chi:
            D[:] = D_test[:]
            chi = chi_test

    # STAGE III
    # Compute partial derivatives dD[:] / dalpha

    # compute perturbed target Q as a function of frequency
    Q_target_pert = np.zeros([len(Q_0), len(f)])

    for n in range(len(Q_0)):
        Q_target_pert[n, :] = Q_0[n] * (f / f_ref) ** (alpha + 0.1)

    # make initial weights
    D_pert = np.ones(N)
    D_pert[:] = D[:]

    # compute initial Q
    chi = 0.0

    for n in np.arange(len(Q_0)):
        A = 1.0
        B = 0.0
        for p in np.arange(N):
            A += tau[n] * (D[p] * w ** 2 * tau_s[p] ** 2) / (
                1.0 + w ** 2 * tau_s[p] ** 2)
            B += tau[n] * (D[p] * w * tau_s[p]) / (
                1.0 + w ** 2 * tau_s[p] ** 2)

        Q = A / B
        chi += sum((Q - Q_target_pert[n, :]) ** 2 / Q_0[n] ** 2)

    #  random search for optimal parameters
    T = T_0

    for _ in xrange(max_it):
        # compute perturbed parameters
        D_test_pert = D_pert * (1.0 + (0.5 - rd.rand(N)) * T)

        # compute test Q
        chi_test = 0.0

        for n in xrange(len(Q_0)):
            A = 1.0
            B = 0.0
            for p in np.arange(N):
                A += tau[n] * (D_test_pert[p] * w ** 2 * tau_s[p] ** 2) / (
                    1.0 + w ** 2 * tau_s[p] ** 2)
                B += tau[n] * (D_test_pert[p] * w * tau_s[p]) / (
                    1.0 + w ** 2 * tau_s[p] ** 2)

            Q_test = A / B
            chi_test += sum((Q_test - Q_target_pert[n, :]) ** 2 / Q_0[n] ** 2)

        # compute new temperature
        T = T * d

        # check if the tested parameters are better
        if chi_test < chi:
            D_pert[:] = D_test_pert[:]
            chi = chi_test

    # sort weights and relaxation times
    decorated = [(tau_s[i], D[i]) for i in range(N)]
    decorated.sort()

    tau_s = [decorated[i][0] for i in range(N)]
    D = [decorated[i][1] for i in range(N)]

    if not quiet:
        print "weights:             ", D
        print "relaxation times:    ", tau_s
        print "partial derivatives: ", (D_pert - D) / 0.1
        print "cumulative rms error:", np.sqrt(chi / (len(Q) * len(Q_0)))

    return np.array(D), np.array(tau_s)


def plot(D_p, tau_p, f_min=None, f_max=None):
    """
    :type D_p: np.ndarray
    :param D_p: The calculated D_p.
    :type tau_p: np.ndarray
    :param tau_p: The calculated tau_p.
    :type f_min: float, optional
    :param f_min: The minimum frequency over which the optimization was
        performed. If given it will be plotted.
    :type f_max: float, optional
    :param f_max: The maximum frequency over which the optimization was
        performed. If given it will be plotted.
    """
    # The Q values to be plotted.
    Q_values = (50, 100, 200, 400)
    # Colors for the corresponding plot.
    Q_colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"]

    f_plot = np.logspace(np.log10(f_min) - 0.2, np.log10(f_max) + 0.2, 100)

    # Angular frequency.
    w_plot = 2.0 * np.pi * f_plot

    # Loop over all desired Q values.
    for target_Q, color in zip(Q_values, Q_colors):
        # Make sure this tau is defined in the same way as the tau in SES3D.
        tau = 1.0 / target_Q

        # compute optimal Q model
        A = 0.0
        B = 0.0
        N = len(D_p)
        for n in xrange(N):
            A = A + D_p[n] * w_plot ** 2 * tau_p[n] ** 2 / \
                (1 + w_plot ** 2 * tau_p[n] ** 2)
            B = B + D_p[n] * w_plot * tau_p[n] / \
                (1 + w_plot ** 2 * tau_p[n] ** 2)

        A = 1 + tau * A
        B = tau * B

        Q_discrete = A / B
        v_discrete = np.sqrt(2 * (A ** 2 + B ** 2) /
                             (A + np.sqrt(A ** 2 + B ** 2)))

        # plot Q and phase velocity as function of frequency
        plt.subplot(221)
        plt.semilogx(f_plot, Q_discrete, "k", linewidth=2, color=color)
        plt.semilogx(f_plot, target_Q * np.ones(len(f_plot)), linewidth=1,
                     linestyle="--", color=color, zorder=-1E6)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Q")
        plt.title("Quality Factor")

        plt.subplot(222)
        plt.semilogx(f_plot, v_discrete, "k", linewidth=2, color=color)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("v")
        plt.title("Phase Velocity")
        plt.grid()

    # Adjust the limits of both plots.
    plt.subplot(221)
    Q_range = max(Q_values) - min(Q_values)
    plt.ylim(min(Q_values) - Q_range / 5.0, max(Q_values) + Q_range / 5.0)
    plt.xlim(f_plot[0], f_plot[-1])

    plt.subplot(222)
    plt.ylim(0.9, 1.1)
    plt.xlim(f_plot[0], f_plot[-1])
    plt.grid()

    # Plot the frequency limits if given.
    if f_min and f_max:
        plt.subplot(221)
        plt.vlines(f_min, *plt.ylim(), color="0.5")
        plt.vlines(f_max, *plt.ylim(), color="0.5")
        plt.text(0.5, 0.98, s="Absorption Band", size="small",
                 transform=plt.gca().transAxes, va="top", ha="center")
        ylim = plt.ylim()
        plt.fill_between(x=[f_min, f_max], y1=[ylim[0], ylim[0]],
                         y2=[ylim[1], ylim[1]], color="0.7", zorder=-2E6)
        plt.subplot(222)
        plt.vlines(f_min, *plt.ylim(), color="0.5")
        plt.vlines(f_max, *plt.ylim(), color="0.5")
        plt.text(0.5, 0.98, s="Absorption Band", size="small",
                 transform=plt.gca().transAxes, va="top", ha="center")
        plt.fill_between(x=[f_min, f_max], y1=[ylim[0], ylim[0]],
                         y2=[ylim[1], ylim[1]], color="0.7", zorder=-2E6)
    plt.gcf().patch.set_alpha(0.0)

    plt.subplot(212)
    dt = min(tau_p) / 10.0
    t = np.arange(0.0, max(tau_p), dt)

    for target_Q, color in zip(Q_values, Q_colors):
        tau = 1.0 / target_Q

        c = np.ones(len(t))

        for n in range(N):
            c += tau * D_p[n] * np.exp(-t / tau_p[n])

        plt.plot(t, c, color=color, label="Q=%i" % target_Q, lw=2)

    plt.xlabel("time [s]")
    plt.grid()
    plt.legend()
    plt.xlim(t[0], t[-1])
    plt.ylabel("C(t)")
    plt.title("Stress Relaxation Functions")
    plt.tight_layout()
