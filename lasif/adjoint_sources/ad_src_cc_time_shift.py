#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An implementation of the time frequency phase misfit and adjoint source after
Fichtner et al. (2008).

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
from scipy.integrate import simps
import obspy.signal.cross_correlation as crosscorr

from lasif import LASIFError


def cc_time_shift(data, synthetic, dt):
    """
    Compute the time shift between two traces by crosscorrelating them.
    :param data: real data
    :param synthetic: synthetic data
    :param dt: Time step
    :return: A value of the time misfit between the two traces.
    """
    # We want to compute how long the shift should be in the crosscorrelation
    # See whether the lengths are the same:
    if not len(data) == len(synthetic):
        raise LASIFError("\n\n Data and Synthetics do not have equal number"
                         " of points. Might be something wrong with your"
                         " processing.")

    # Possibly try to adjust the shift length so its more stable for small
    # windows.
    length = len(np.nonzero(synthetic))
    cc = crosscorr.correlate(a=synthetic, b=data, shift=length)

    shift = np.argmax(cc) - length  # Correct shift in the cross_corr
    time_shift = dt * shift

    return time_shift


def adsrc_cc_time_shift(t, data, synthetic, min_period, max_period,
                        plot=False, colorbar_axis=None):
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
    # Compute time shift between the two traces.
    time_shift = cc_time_shift(data, synthetic, dt)
    messages.append(f"time shift was {time_shift} seconds")

    # Now we have the time shift. We need velocity of synthetics.
    # vel_syn = np.gradient(synthetic, x=t)
    vel_syn = np.diff(synthetic) / dt
    # vel_syn = np.append(vel_syn, 0.0)
    norm_const = simps(np.square(vel_syn), dx=dt)
    # norm_const = np.sum(np.square(vel_syn)) * (t[1]-t[0])
    ad_src = (time_shift / norm_const) * vel_syn * dt
    # orig_length = len(ad_src)
    #
    # # Find first non-zero value and last non-zero value. Trim, taper, pad.
    # n_zero = np.nonzero(ad_src)
    # pad = int(min_period / dt)
    # start = n_zero[0][0] - pad
    # end = n_zero[0][-1] + pad
    # print(f"start: {start}")
    # print(f"end: {end}")
    # print(f"pad: {pad}")
    # ad_src = np.trim_zeros(ad_src)
    # ad_src = np.concatenate([np.zeros(pad), ad_src, np.zeros(pad)])
    # ad_src = obspy.Trace(ad_src)
    # ad_src.stats.delta = dt
    # ad_src = ad_src.filter("bandpass", freqmin=1 / max_period,
    #                                     freqmax=1 / min_period,
    #                                     zerophase = True, corners=2).data
    # ad_src = obspy.Trace(ad_src).taper(max_percentage=0.05, type="hann").data
    # front_pad = np.zeros(start)
    # back_pad = np.zeros(orig_length-end - 1)
    # ad_src = np.concatenate([front_pad, ad_src, back_pad])
    #
    ad_src = ad_src[::-1]
    ad_src = np.concatenate([[0.0], ad_src])

    ret_dict = {"adjoint_source": ad_src,
                "misfit_value": time_shift,
                "details": {"messages": messages}}

    if plot:
        import matplotlib.pyplot as plt
        # plt.style.use("seaborn-whitegrid")
        # from lasif.colors import get_colormap
        # fig = plt.gcf()

        plt.subplot(212)
        plt.plot(t, synthetic, color="red")
        plt.legend(["Synthetics"])

        plt.subplot(211)
        plt.plot(t, ad_src[::-1], color="black")
        plt.legend(["CrossCorrelation Adjoint Source"])

        # plt.plot(t, synthetic, color="red", linestyle='--')
        # plt.legend(["Adjoint Source", "Synthetics"])

    return ret_dict
