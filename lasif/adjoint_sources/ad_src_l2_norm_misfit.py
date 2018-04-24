#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple L2-norm misfit.

:copyright:
    Solvi Thrastarson (soelvi.thrastarson@erdw.ethz.ch)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from obspy.signal.invsim import cosine_taper
from scipy.integrate import simps


def adsrc_l2_norm_misfit(t, data, synthetic, min_period, max_period,
                         plot=False):
    """
    Calculates the L2-norm misfit and adjoint source.

    :type data: np.ndarray
    :param data: The measured data array
    :type synthetic: np.ndarray
    :param synthetic: The calculated data array
    :param plot: boolean value deciding whether to plot or not
    :type plot: boolean
    :rtype: dictionary
    :returns: Return a dictionary with three keys:
        * adjoint_source: The calculated adjoint source as a numpy array
        * misfit: The misfit value
        * messages: A list of strings giving additional hints to what happened
            in the calculation.
    """
    messages = []
    if len(synthetic) != len(data):
        msg = "Both arrays need to have equal length"
        raise ValueError(msg)
    dt = t[1] - t[0]
    diff = synthetic - data
    diff = np.require(diff, dtype="float64")
    l2norm = 0.5 * simps(np.square(diff), dx=dt)
    ad_src = diff * dt

    orig_length = len(ad_src)
    n_zeros = np.nonzero(ad_src)
    ad_src = np.trim_zeros(ad_src)
    len_window = len(ad_src) * dt
    messages.append(f"Length of window is: {len_window} seconds")
    ratio = min_period * 2 / len_window
    p = ratio / 2.0  # We want the minimum window to taper 25% off each side
    if p > 1.0:  # For manually picked small windows.
        p = 1.0
    window = cosine_taper(len(ad_src), p=p)
    ad_src = ad_src * window
    front_pad = np.zeros(n_zeros[0][0])
    back_pad = np.zeros(orig_length - n_zeros[0][-1] - 1)
    ad_src = np.concatenate([front_pad, ad_src, back_pad])

    adjoint_source = ad_src[::-1]  # This might need to be polarized

    ret_dict = {
        "adjoint_source": adjoint_source,
        "misfit_value": l2norm,
        "details": {"messages": messages}}

    if plot:
        adjoint_source_plot(t, data, synthetic, adjoint_source, l2norm)

    return ret_dict


def adjoint_source_plot(t, data, synthetic, adjoint_source, misfit):

    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(t, data, color="0.2", label="Data", lw=2)
    plt.plot(t, synthetic, color="#bb474f",
             label="Synthetic", lw=2)

    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.subplot(212)
    plt.plot(t, adjoint_source[::-1], color="#2f8d5b", lw=2,
             label="Adjoint Source")
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.title(f"L2Norm Adjoint Source with a Misfit of {misfit}")
