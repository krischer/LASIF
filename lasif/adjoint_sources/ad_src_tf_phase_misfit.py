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
import warnings

import numexpr as ne
import numpy as np
import obspy
from obspy.signal.interpolation import lanczos_interpolation


from lasif import LASIFAdjointSourceCalculationError
from lasif.adjoint_sources import time_frequency, utils

eps = np.spacing(1)


def adsrc_tf_phase_misfit(t, data, synthetic, min_period, max_period,
                          plot=False, max_criterion=7.0):
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
    #
    # The current choice is historic as changing does (very slightly) chance
    # the calculated misfit and we don't want to disturb inversions in
    # progress. The difference is likely minimal in any case. We might have
    # same aliasing into the lower frequencies but the filters coupled with
    # the TF-domain weighting will get rid of them in essentially all
    # realistically occurring cases.
    dt_new = max(float(int(min_period / 3.0)), t[1] - t[0])

    # New time axis
    ti = utils.matlab_range(t[0], t[-1], dt_new)
    # Make sure its odd - that avoid having to deal with some issues
    # regarding frequency bin interpolation. Now positive and negative
    # frequencies will always be all symmetric. Data is assumed to be
    # tapered in any case so no problem are to be expected.
    if not len(ti) % 2:
        ti = ti[:-1]

    # Interpolate both signals to the new time axis - this massively speeds
    # up the whole procedure as most signals are highly oversampled. The
    # adjoint source at the end is re-interpolated to the original sampling
    # points.
    original_data = data
    original_synthetic = synthetic
    data = lanczos_interpolation(
        data=data, old_start=t[0], old_dt=t[1] - t[0], new_start=t[0],
        new_dt=dt_new, new_npts=len(ti), a=8, window="blackmann")
    synthetic = lanczos_interpolation(
        data=synthetic, old_start=t[0], old_dt=t[1] - t[0], new_start=t[0],
        new_dt=dt_new, new_npts=len(ti), a=8, window="blackmann")
    original_time = t
    t = ti

    # -------------------------------------------------------------------------
    # Compute time-frequency representations

    # Window width is twice the minimal period.
    width = 2.0 * min_period

    # Compute time-frequency representation of the cross-correlation
    _, _, tf_cc = time_frequency.time_frequency_cc_difference(
        t, data, synthetic, width)
    # Compute the time-frequency representation of the synthetic
    tau, nu, tf_synth = time_frequency.time_frequency_transform(t, synthetic,
                                                                width)

    # -------------------------------------------------------------------------
    # compute tf window and weighting function

    # noise taper: down-weight tf amplitudes that are very low
    tf_cc_abs = np.abs(tf_cc)
    m = tf_cc_abs.max() / 10.0  # NOQA
    weight = ne.evaluate("1.0 - exp(-(tf_cc_abs ** 2) / (m ** 2))")

    nu_t = nu.T

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
    abs_weighted_DP = np.abs(weight * DP)
    _x = abs_weighted_DP.max()  # NOQA
    test_field = ne.evaluate("weight * DP / _x")

    criterion_1 = np.sum([np.abs(np.diff(test_field, axis=0)) > 0.7])
    criterion_2 = np.sum([np.abs(np.diff(test_field, axis=1)) > 0.7])
    criterion = np.sum([criterion_1, criterion_2])
    # Compute the phase misfit
    dnu = nu[1] - nu[0]

    i = ne.evaluate("sum(weight ** 2 * DP ** 2)")

    phase_misfit = np.sqrt(i * dt_new * dnu)

    # Sanity check. Should not occur.
    if np.isnan(phase_misfit):
        msg = "The phase misfit is NaN."
        raise LASIFAdjointSourceCalculationError(msg)

    # The misfit can still be computed, even if not adjoint source is
    # available.
    if criterion > max_criterion:
        warning = ("Possible phase jump detected. Misfit included. No "
                   "adjoint source computed. Criterion: %.1f - Max allowed "
                   "criterion: %.1f" % (criterion, max_criterion))
        warnings.warn(warning)
        messages.append(warning)

        ret_dict = {
            "adjoint_source": None,
            "misfit_value": phase_misfit,
            "details": {"messages": messages}
        }

        return ret_dict

    # Make kernel for the inverse tf transform
    idp = ne.evaluate(
        "weight ** 2 * DP * tf_synth / (m + abs(tf_synth) ** 2)")

    # Invert tf transform and make adjoint source
    ad_src, it, I = time_frequency.itfa(tau, idp, width)

    # Interpolate both signals to the new time axis
    ad_src = lanczos_interpolation(
        # Pad with a couple of zeros in case some where lost in all
        # these resampling operations. The first sample should not
        # change the time.
        data=np.concatenate([ad_src.imag, np.zeros(100)]),
        old_start=tau[0],
        old_dt=tau[1] - tau[0],
        new_start=original_time[0],
        new_dt=original_time[1] - original_time[0],
        new_npts=len(original_time), a=8, window="blackmann")

    # Divide by the misfit and change sign.
    ad_src /= (phase_misfit + eps)
    ad_src = -1.0 * np.diff(ad_src) / (t[1] - t[0])

    # Taper at both ends. Exploit ObsPy to not have to deal with all the
    # nasty things.
    ad_src = \
        obspy.Trace(ad_src).taper(max_percentage=0.05, type="hann").data

    # Reverse time and add a leading zero so the adjoint source has the
    # same length as the input time series.
    ad_src = ad_src[::-1]
    ad_src = np.concatenate([[0.0], ad_src])

    # Plot if requested. ------------------------------------------------------
    if plot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-whitegrid")
        from lasif.colors import get_colormap

        if isinstance(plot, mpl.figure.Figure):
            fig = plot
        else:
            fig = plt.gcf()

        # Manually set-up the axes for full control.
        l, b, w, h = 0.1, 0.05, 0.80, 0.22
        rect = l, b + 3 * h, w, h
        waveforms_axis = fig.add_axes(rect)
        rect = l, b + h, w, 2 * h
        tf_axis = fig.add_axes(rect)
        rect = l, b, w, h
        adj_src_axis = fig.add_axes(rect)
        rect = l + w + 0.02, b, 1.0 - (l + w + 0.02) - 0.05, 4 * h
        cm_axis = fig.add_axes(rect)

        # Plot the weighted phase difference.
        weighted_phase_difference = (DP * weight).transpose()
        mappable = tf_axis.pcolormesh(
            tau, nu, weighted_phase_difference, vmin=-1.0, vmax=1.0,
            cmap=get_colormap("tomo_full_scale_linear_lightness_r"),
            shading="gouraud", zorder=-10)
        tf_axis.grid(True)
        tf_axis.grid(True, which='minor', axis='both', linestyle='-',
                     color='k')

        cm = fig.colorbar(mappable, cax=cm_axis)
        cm.set_label("Phase difference in radian", fontsize="large")

        # Various texts on the time frequency domain plot.
        text = "Misfit: %.4f" % phase_misfit
        tf_axis.text(x=0.99, y=0.02, s=text, transform=tf_axis.transAxes,
                     fontsize="large", color="#C25734", fontweight=900,
                     verticalalignment="bottom",
                     horizontalalignment="right")

        txt = "Weighted Phase Difference - red is a phase advance of the " \
              "synthetics"
        tf_axis.text(x=0.99, y=0.95, s=txt,
                     fontsize="large", color="0.1",
                     transform=tf_axis.transAxes,
                     verticalalignment="top",
                     horizontalalignment="right")

        if messages:
            message = "\n".join(messages)
            tf_axis.text(x=0.99, y=0.98, s=message,
                         transform=tf_axis.transAxes,
                         bbox=dict(facecolor='red', alpha=0.8),
                         verticalalignment="top",
                         horizontalalignment="right")

        # Adjoint source.
        adj_src_axis.plot(original_time, ad_src[::-1], color="0.1", lw=2,
                          label="Adjoint source (non-time-reversed)")
        adj_src_axis.legend()

        # Waveforms.
        waveforms_axis.plot(original_time, original_data, color="0.1", lw=2,
                            label="Observed")
        waveforms_axis.plot(original_time, original_synthetic,
                            color="#C11E11", lw=2, label="Synthetic")
        waveforms_axis.legend()

        # Set limits for all axes.
        tf_axis.set_ylim(0, 2.0 / min_period)
        tf_axis.set_xlim(0, tau[-1])
        adj_src_axis.set_xlim(0, tau[-1])
        waveforms_axis.set_xlim(0, tau[-1])

        waveforms_axis.set_ylabel("Velocity [m/s]", fontsize="large")
        tf_axis.set_ylabel("Period [s]", fontsize="large")
        adj_src_axis.set_xlabel("Seconds since event", fontsize="large")

        # Hack to keep ticklines but remove the ticks - there is probably a
        # better way to do this.
        waveforms_axis.set_xticklabels([
            "" for _i in waveforms_axis.get_xticks()])
        tf_axis.set_xticklabels(["" for _i in tf_axis.get_xticks()])

        _l = tf_axis.get_ylim()
        _r = _l[1] - _l[0]
        _t = tf_axis.get_yticks()
        _t = _t[(_l[0] + 0.1 * _r < _t) & (_t < _l[1] - 0.1 * _r)]

        tf_axis.set_yticks(_t)
        tf_axis.set_yticklabels(["%.1fs" % (1.0 / _i) for _i in _t])

        waveforms_axis.get_yaxis().set_label_coords(-0.08, 0.5)
        tf_axis.get_yaxis().set_label_coords(-0.08, 0.5)

        fig.suptitle("Time Frequency Phase Misfit and Adjoint Source",
                     fontsize="xx-large")

    ret_dict = {
        "adjoint_source": ad_src,
        "misfit_value": phase_misfit,
        "details": {"messages": messages}
    }

    return ret_dict
