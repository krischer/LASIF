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
    # Move the maximum period in each direction to avoid cycle skip.
    shift = int(min_period / dt)
    # Compute time shift between the two traces.
    time_shift = cc_time_shift(data, synthetic, dt, shift)
    messages.append(f"time shift was {time_shift} seconds")

    # Now we have the time shift. We need velocity of synthetics.
    vel_syn = np.diff(synthetic) / dt
    norm_const = simps(np.square(vel_syn), dx=dt)
    ad_src = (time_shift / norm_const) * vel_syn * dt

    ad_src = ad_src[::-1]  # Time reverse
    ad_src = np.concatenate([[0.0], ad_src])  # Add a zero lost in the diff

    ret_dict = {"adjoint_source": ad_src,
                "misfit_value": time_shift,
                "details": {"messages": messages}}

    if plot:
        import matplotlib.pyplot as plt

        plt.subplot(212)
        plt.plot(t, synthetic, color="red")
        plt.plot(t, data, color="blue")
        plt.legend(["Synthetics", "Data"])

        plt.subplot(211)
        plt.plot(t, ad_src[::-1], color="black")
        plt.legend(["CrossCorrelation Adjoint Source"])

    return ret_dict
