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
from obspy.signal.interpolation import lanczos_interpolation
import scipy.signal
from scipy.interpolate import RectBivariateSpline
import warnings


from lasif import LASIFAdjointSourceCalculationError
from lasif.adjoint_sources import time_frequency, utils

eps = np.spacing(1)


def adsrc_tf_phase_misfit(t, data, synthetic, min_period, max_period,
                          axis=None, colorbar_axis=None):
    """
    :rtype: dictionary
    :returns: Return a dictionary with three keys:
        * adjoint_source: The calculated adjoint source as a numpy array
        * misfit: The misfit value
        * messages: A list of strings giving additional hints to what happened
            in the calculation.
    """
    # Assumes that t starts at 0. Pad your data if that is not the case -
    # Parts with zeros are essentially skipped making it fairly efficient.
    assert t[0] == 0

    messages = []

    # Internal sampling interval. Some explanations for this "magic" number.
    # LASIF's preprocessing allows no frequency content with smaller periods
    # than min_period / 2.2 (see function_templates/preprocesssing_function.py
    # for details). Assuming most users don't change this, this is equal to
    # the Nyquist frequency and the largest possible sampling interval to
    # catch everything is min_period / 4.4.
    dt_new = min_period / 4.4


    # Downsample data and synthetics - they are usually way oversampled in
    # any case and this greatly speeds up the calculations.

    # New time axis
    ti = utils.matlab_range(t[0], t[-1], dt_new)
    # Interpolate both signals to the new time axis
    data = lanczos_interpolation(
        data=data, old_start=t[0], old_dt=t[1] - t[0], new_start=t[0],
        new_dt=dt_new, new_npts=len(ti), a=8, window="blackmann")
    synthetic = lanczos_interpolation(
        data=synthetic, old_start=t[0], old_dt=t[1] - t[0], new_start=t[0],
        new_dt=dt_new, new_npts=len(ti), a=8, window="blackmann")
    original_time = t
    t = ti


    # Compute time-frequency representations ----------------------------------

    # Window width is twice the minimal period.
    width = 2.0 * min_period

    # Compute time-frequency representation of the cross-correlation
    _, _, tf_cc = time_frequency.time_frequency_cc_difference(
        t, data, synthetic, width)
    # Compute the time-frequency representation of the synthetic
    tau, nu, tf_synth = time_frequency.time_frequency_transform(t, synthetic,
                                                                width)

    # compute tf window and weighting function --------------------------------

    # noise taper: downweigh tf amplitudes that are very low
    m = np.abs(tf_cc).max() / 10.0
    weight = 1.0 - np.exp(-(np.abs(tf_cc) ** 2) / (m ** 2))
    nu_t = nu.transpose()

    # highpass filter (periods longer than max_period are suppressed
    # exponentially)
    weight *= (1.0 - np.exp(-(nu_t * max_period) ** 2))

    # lowpass filter (periods shorter than min_period are suppressed
    # exponentially)
    nu_t_large = np.zeros(nu_t.shape)
    nu_t_small = np.zeros(nu_t.shape)
    thres = (nu_t <= 1.0 / min_period)
    nu_t_large[np.invert(thres)] = 1.0
    nu_t_small[thres] = 1.0
    weight *= (np.exp(-10.0 * np.abs(nu_t * min_period - 1.0)) * nu_t_large +
               nu_t_small)

    # normalisation
    weight /= weight.max()

    # computation of phase difference, make quality checks and misfit ---------

    # Compute the phase difference.
    # DP = np.imag(np.log(m + tf_cc / (2 * m + np.abs(tf_cc))))
    DP = np.angle(tf_cc)

    # Attempt to detect phase jumps by taking the derivatives in time and
    # frequency direction. 0.7 is an emperical value.
    test_field = weight * DP / np.abs(weight * DP).max()
    criterion_1 = np.sum([np.abs(np.diff(test_field, axis=0)) > 0.7])
    criterion_2 = np.sum([np.abs(np.diff(test_field, axis=1)) > 0.7])
    criterion = np.sum([criterion_1, criterion_2])
    # criterion_1 = np.abs(np.diff(test_field, axis=0)).max()
    # criterion_2 = np.abs(np.diff(test_field, axis=1)).max()
    # criterion = max(criterion_1, criterion_2)
    if criterion > 7.0:
        warning = ("Possible phase jump detected. Misfit included. No "
                   "adjoint source computed.")
        warnings.warn(warning)
        messages.append(warning)

    # Compute the phase misfit
    dnu = nu[1] - nu[0]
    phase_misfit = np.sqrt(np.sum(weight ** 2 * DP ** 2) * dt_new * dnu)

    # Sanity check. Should not occur.
    if np.isnan(phase_misfit):
        msg = "The phase misfit is NaN."
        raise LASIFAdjointSourceCalculationError(msg)

    # compute the adjoint source when no phase jump detected ------------------

    if criterion <= 7.0:
        # Make kernel for the inverse tf transform
        idp = weight * weight * DP * tf_synth / (m + np.abs(tf_synth) *
                                                 np.abs(tf_synth))

        # Invert tf transform and make adjoint source
        ad_src, it, I = time_frequency.itfa(tau, idp, width)

        # Interpolate to original time axis
        current_time = tau
        new_time = original_time[original_time <= current_time.max()]

        # Interpolate both signals to the new time axis
        ad_src = lanczos_interpolation(
            data=np.require(ad_src.imag, dtype=np.float64, requirements=["C"]),
            old_start=current_time[0],
            old_dt=current_time[1] - current_time[0],
            new_start=new_time[0],
            new_dt=new_time[1] - new_time[0],
            new_npts=len(new_time), a=8, window="blackmann")

        if len(t) > len(new_time):
            ad_src = np.concatenate([ad_src, np.zeros(len(t) - len(new_time))])

        # Divide by the misfit and change sign.
        ad_src /= (phase_misfit + eps)
        ad_src = -1.0 * np.diff(ad_src) / (t[1] - t[0])

        # Reverse time and add a leading zero so the adjoint source has the
        # same length as the input time series.
        ad_src = ad_src[::-1]
        ad_src = np.concatenate([[0.0], ad_src])

    else:
        # Criterion failed, no misfit and adjoint source calculated.
        raise LASIFAdjointSourceCalculationError(
            "Criterion failed, no misfit has been calculated.")

    # Plot if required. -------------------------------------------------------

    if axis:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        # Primary axis: plot weighted phase difference. -----------------------

        weighted_phase_difference = (DP * weight).transpose()
        abs_diff = np.abs(weighted_phase_difference)
        mappable = axis.pcolormesh(tau, nu, weighted_phase_difference,
                                   vmin=-1.0, vmax=1.0, cmap=cm.RdBu_r)
        axis.set_xlabel("Seconds since event")
        axis.set_ylabel("Frequency [Hz]")

        # Smart scaling for the frequency axis.
        temp = abs_diff.max(axis=1) * (nu[1] - nu[0])
        ymax = len(temp[temp > temp.max() / 1000.0])
        ymax *= nu[1, 0] - nu[0, 0]
        ymax *= 2
        axis.set_ylim(0, 2.0 / min_period)

        if colorbar_axis:
            cm = plt.gcf().colorbar(mappable, cax=colorbar_axis)
        else:
            cm = plt.gcf().colorbar(mappable, ax=axis)
        cm.set_label("Phase difference in radian")

        # Secondary axis: plot waveforms and adjoint source. ------------------

        ax2 = axis.twinx()

        ax2.plot(t, ad_src, color="black", alpha=1.0)
        min_value = min(ad_src.min(), -1.0)
        max_value = max(ad_src.max(), 1.0)

        value_range = max_value - min_value
        axis.twin_axis = ax2
        ax2.set_ylim(min_value - 2.5 * value_range,
                     max_value + 0.5 * value_range)
        axis.set_xlim(0, tau[:, -1][-1])
        ax2.set_xlim(0, tau[:, -1][-1])
        ax2.set_yticks([])

        text = "Misfit: %.4f" % phase_misfit
        axis.text(x=0.99, y=0.02, s=text, transform=axis.transAxes,
                  bbox=dict(facecolor='orange', alpha=0.8),
                  verticalalignment="bottom",
                  horizontalalignment="right")

        if messages:
            message = "\n".join(messages)
            axis.text(x=0.99, y=0.98, s=message, transform=axis.transAxes,
                      bbox=dict(facecolor='red', alpha=0.8),
                      verticalalignment="top",
                      horizontalalignment="right")

    ret_dict = {
        "adjoint_source": ad_src,
        "misfit_value": phase_misfit,
        "details": {"messages": messages}
    }

    return ret_dict
