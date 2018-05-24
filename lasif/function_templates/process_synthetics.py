#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project specific function for modifying synthetics on the fly.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import copy

def process_synthetics(st, processing_params, event):  # NOQA
    """
    This function is called after a synthetic file has been read.

    Do whatever you need to do in here an return a potentially modified
    stream object. Make sure that anything you do works with the
    processing function. LASIF expects data and synthetics to have
    exactly the same length before it can pick windows and calculate adjoint
    sources.

    Potential uses for this function are to shift synthetics in time if
    required or to apply some processing to them which LASIF by default does
    not do.

    Please note that you also got the iteration object here, so if you
    want some parameters to change depending on the iteration, just use
    if/else on the iteration objects.
    """

    min_period = processing_params['highpass_period']
    max_period = processing_params['lowpass_period']
    st = copy.deepcopy(st)  # We do not want to modify actual synthetics
    # Currently a no-op.
    # This function will modify each waveform stream. It must
    # be called process() and it takes three arguments:
    #
    # * st: The obspy.Stream object with the waveforms.
    # * inv: The obspy.Inventory object with the metadata.
    # * tag: The name of the currently selected tag.

    # Assuming displacement seismograms
    for tr in st:
        tr.stats.starttime = \
            event["origin_time"] + processing_params["salvus_start_time"]

    if processing_params["stf"] == "heaviside":

        # Bandpass filtering
        st.detrend("linear")
        st.detrend("demean")
        st.taper(0.05, type="cosine")
        st.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=False)

        st.detrend("linear")
        st.detrend("demean")
        st.taper(0.05, type="cosine")
        st.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=False)
    # for tr in st:
    #     tr.data = np.require(tr.data, dtype="float32", requirements="C")

    return st
