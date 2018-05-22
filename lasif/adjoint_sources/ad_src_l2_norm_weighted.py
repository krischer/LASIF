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
from lasif import LASIFNotFoundError


def find_envelope(station, asdf_file):
    """
    Find the corresponding envelope needed for the adjoint and misfit
    calculation
    :param station: station which the window is being calculated for
    :param asdf_file: the asdf file needed
    :return: weighting_function: The envelope which is used for weighting
    """
    import pyasdf
    with pyasdf.ASDFDataSet(asdf_file, mode="r") as db:
        if "BinEnvelope" not in db.auxiliary_data.list():
            msg = "You have yet to create your station bins and envelopes.\n "\
                  "Run lasif get_weighting_bins --help for more information."
            raise LASIFNotFoundError(msg)

        envelopes = db.auxiliary_data["BinEnvelope"]
        if "_" in station:
            components = station.split("_")
            station = ".".join(components)

        found = False
        for envelope in envelopes:
            if station in envelope.parameters["stations"]:
                weighting_function = np.array(envelope.data)
                found = True
        if not found:
            raise ValueError(f"Did not find station {station} in "
                             f"the available envelopes.")

        eps = 0.1
        weighting_function += eps

        return weighting_function


def adsrc_l2_norm_weighted(t, data, synthetic, min_period,
                           event, station, envelope, plot=False):
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

    # Multiply data and synthetics with the inverse of the envelope
    # Find corresponding envelope.
    # tag = commi.waveforms.preprocessing_tag
    # asdf_file = commi.waveforms.get_asdf_filename(event_name=event,
    #                                              data_type="processed",
    #                                              tag_or_iteration=tag)
    # if not os.path.exists(asdf_file):
    #     msg = f"{asdf_file} does not exist. Make sure you have " \
    #           f"the correct values for highpass and lowpass " \
    #           f"periods in your config file."
    #     raise LASIFNotFoundError(msg)

    weight = envelope
    data /= weight
    synthetic /= weight

    diff = data - synthetic
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
    ad_src /= weight

    # Weigh the adjoint source by the

    adjoint_source = ad_src[::-1]  # This might need to be polarized

    ret_dict = {
        "adjoint_source": adjoint_source,
        "misfit_value": l2norm,
        "details": {"messages": messages}}

    if plot:
        adjoint_source_plot(t, data, synthetic, adjoint_source, l2norm, weight)

    return ret_dict


def adjoint_source_plot(t, data, synthetic, adjoint_source, misfit, envelope):

    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(t, data, color="0.2", label="Data", lw=2)
    plt.plot(t, synthetic, color="#bb474f",
             label="Synthetic", lw=2)
    plt.plot(t, envelope * (max(data) / max(envelope)), color="blue")

    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.subplot(212)
    plt.plot(t, adjoint_source[::-1], color="#2f8d5b", lw=2,
             label="Adjoint Source")
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.title(f"L2Norm Adjoint Source with a Misfit of {misfit}")
