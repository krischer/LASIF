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
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
import warnings

from lasif.adjoint_sources import time_frequency

eps = np.spacing(1)


def adsrc_tf(t, data, synthetic, dt_new, width, threshold):
    """
    """
    messages = []
    # Compute time-frequency representation via cross-correlation
    tau_cc, nu_cc, tf_cc = time_frequency.time_frequency_cc_difference(
        t, data, synthetic, dt_new, width, threshold)
    # Compute the time-frequency representation of the synthetic
    tau, nu, tf_synth = time_frequency.time_frequency_transform(t,
        synthetic, dt_new, width, threshold)

    # 2D interpolation. Use a two step interpolation for the real and the
    # imaginary parts.
    tf_cc_interp = RectBivariateSpline(tau_cc[0], nu_cc[:, 0], tf_cc.real,
        kx=1, ky=1, s=0)(tau[0], nu[:, 0])
    tf_cc_interp = np.require(tf_cc_interp, dtype="complex128")
    tf_cc_interp.imag = RectBivariateSpline(tau_cc[0], nu_cc[:, 0], tf_cc.imag,
        kx=1, ky=1, s=0)(tau[0], nu[:, 0])
    tf_cc = tf_cc_interp

    # Make window functionality
    # noise taper
    m = np.abs(tf_cc).max() / 10.0
    weight = 1.0 - np.exp(-(np.abs(tf_cc) ** 2) / (m ** 2))
    nu_t = nu.transpose()
    # high-pass filter
    weight *= (1.0 - np.exp((-nu_t ** 2) / (0.002 ** 2)))
    nu_t_large = np.zeros(nu_t.shape)
    nu_t_small = np.zeros(nu_t.shape)
    thres = (nu_t <= 0.005)
    nu_t_large[np.invert(thres)] = 1.0
    nu_t_small[thres] = 1.0
    # low-pass filter
    weight *= (np.exp(-(nu_t - 0.005) ** 4 / 0.005 ** 4) *
        nu_t_large + nu_t_small)
    # normalisation
    weight /= weight.max()

    # Compute the phase difference.
    DP = np.imag(np.log(eps + tf_cc / (eps + np.abs(tf_cc))))

    # Attempt to detect phase jumps by taking the derivatives in time and
    # frequency direction. 0.7 is an emperical value.
    test_field = weight * DP / np.abs(weight * DP).max()
    criterion_1 = np.abs(np.diff(test_field, axis=0)).max()
    criterion_2 = np.abs(np.diff(test_field, axis=1)).max()
    criterion = max(criterion_1, criterion_2)
    if criterion > 0.7:
        warning = "Possible phase jump detected"
        warnings.warn(warning)
        messages.append(warning)

    # Compute the phase misfit
    dnu = nu[1, 0] - nu[0, 0]
    phase_misfit = np.sqrt(np.sum(weight ** 2 * DP ** 2) * dt_new * dnu)

    # Sanity check. Should not occur.
    if np.isnan(phase_misfit):
        msg = "The phase misfit is NaN."
        raise Exception(msg)

    # Make kernel for the inverse tf transform
    idp = weight * weight * DP * tf_synth / (eps + np.abs(tf_synth) *
        np.abs(tf_synth))

    # Invert tf transform and make adjoint source
    ad_src, it, I = time_frequency.itfa(tau, nu, idp, width, threshold)

    # Interpolate to original time axis
    ad_src = interp1d(tau[0, :], np.imag(ad_src), kind=2)(t)

    # Divide by the misfit.
    ad_src /= (phase_misfit + eps)
    ad_src = np.diff(ad_src) / (t[1] - t[0])

    # Reverse time and add a leading zero so the adjoint source has the same
    # length as the input time series.
    ad_src = ad_src[::-1]
    ad_src = np.concatenate([[0.0], ad_src])

    ret_dict = {
        "adjoint_source": ad_src,
        "misfit": phase_misfit,
        "messages": messages}

    return ret_dict
