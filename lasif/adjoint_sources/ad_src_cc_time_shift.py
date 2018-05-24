#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An implementation of the cross correlation time shift adjoint source.

:copyright:
    Solvi Thrastarson (soelvi.thrastarson@erdw.ethz.ch)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
from scipy.integrate import simps
from obspy.signal.invsim import cosine_taper
import obspy.signal.cross_correlation as crosscorr
from lasif import LASIFError


def cc_time_shift(data, synthetic, dt, shift):
    """
    Compute the time shift between two traces by crosscorrelating them.
    :param data: real data
    :param synthetic: synthetic data
    :param dt: Time step
    :param shift: How far to shift to both directions in the crosscorr
    :return: A value of the time misfit between the two traces.
    """
    # See whether the lengths are the same:
    if not len(data) == len(synthetic):
        raise LASIFError("\n\n Data and Synthetics do not have equal number"
                         " of points. Might be something wrong with your"
                         " processing.")

    cc = crosscorr.correlate(a=synthetic, b=data, shift=shift)

    shift = np.argmax(cc) - shift  # Correct shift in the cross_corr
    time_shift = dt * shift

    return time_shift


def adsrc_cc_time_shift(t, data, synthetic, min_period, max_period,
                        plot=False):
    """
       :rtype: dictionary
       :returns: Return a dictionary with three keys:
           * adjoint_source: The calculated adjoint source as a numpy array
           * misfit: The misfit value
           * messages: A list of strings giving additional hints to what
                happened in the calculation.
    """

    messages = []

    dt = t[1] - t[0]

    # Move the minimum period in each direction to avoid cycle skip.
    shift = int(min_period / dt)  # This can be adjusted if you wish so.

    # Compute time shift between the two traces.
    time_shift = cc_time_shift(data, synthetic, dt, shift)
    misfit = 1.0 / 2.0 * time_shift ** 2
    messages.append(f"Time shift was {time_shift} seconds")
    if np.abs(time_shift) > (min_period / 4):
        messages.append(f"Time shift too big for adjoint source calculation, "
                        f"we will only return misfit")
        if plot:
            print("Time shift too large to calculate an adjoint source. "
                  "Misfit included though")
        ad_src = np.zeros(len(t))
        ret_dict = {"adjoint_source": ad_src,
                    "misfit_value": misfit,
                    "details": {"messages": messages}}
        return ret_dict

    lag = int(abs(time_shift) / dt)
    d_vel = np.gradient(data) / dt
    if time_shift < 0:
        d_vel = np.roll(d_vel.data, lag)
    elif time_shift > 0:
        d_vel = np.roll(d_vel.data, -lag)

    # Now we have the time shift. We need velocity of synthetics.
    vel_syn = np.gradient(synthetic) / dt
    norm_const = simps(vel_syn * d_vel, dx=dt)
    ad_src = (time_shift / norm_const) * d_vel * dt
    orig_length = len(ad_src)

    n_zeros = np.nonzero(ad_src)
    ad_src = np.trim_zeros(ad_src)
    len_window = len(ad_src) * dt

    # return an empty adjoint source if window is too short to be meaningful
    if len_window < 2 * min_period:
        warning = "Window length to short to compute a meaningful misfit" \
                  "and adjoint source"
        messages.append(warning)
        misfit = 0.0
        ret_dict = {"adjoint_source": np.zeros_like(data),
                    "misfit_value": misfit,
                    "details": {"messages": messages}}
        return ret_dict

    messages.append(f"Length of window is: {len_window} seconds")

    # Taper the adjoint source again
    ratio = min_period * 2.0 / len_window
    p = ratio / 2.0  # We want the minimum window to taper 25% off each side
    if p > 1.0:  # For manually picked small windows.
        p = 1.0
    window = cosine_taper(len(ad_src), p=p)
    ad_src = ad_src * window
    front_pad = np.zeros(n_zeros[0][0])
    back_pad = np.zeros(orig_length - n_zeros[0][-1] - 1)
    ad_src = np.concatenate([front_pad, ad_src, back_pad])
    ad_src = ad_src[::-1]  # Time reverse

    ret_dict = {"adjoint_source": ad_src,
                "misfit_value": misfit,
                "details": {"messages": messages}}

    if plot:
        adjoint_source_plot(t, data, synthetic, ad_src, misfit, time_shift)

    return ret_dict


def adjoint_source_plot(t, data, synthetic, adjoint_source, misfit,
                        time_shift):

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

    plt.title(f"CCTimeShift Adjoint Source with a Misfit of {misfit}. "
              f"Time shift {time_shift}")
