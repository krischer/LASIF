#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project specific source time function.

Only relevant when SES3D is used as a solver as SPECFEM always uses a
delta-like source time function.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
import obspy


def source_time_function(npts, delta, freqmin, freqmax, iteration):
    """
    Source time function used for simulating with SES3D.

    Do whatever you wish in here but make sure it returns an array of
    ``npts`` samples. Furthermore the spectral content of source time
    function + processing of synthetics should equal the spectral content of
    the processed data. Otherwise the seismograms cannot readily be compared.

    The first sample in the resulting source time function also has to be zero!

    Freqmin and freqmax specify the bandpass filter frequencies.

    :param npts: The desired number of samples.
    :param delta: The sample spacing.
    :param freqmin: The minimum desired frequency.
    :param freqmax: The maximum desired frequency.
    :param iteration: The current iteration.

    Please note that you also got the iteration object here, so if you
    want some parameters to change depending on the iteration, just use
    if/else on the iteration objects.

    >>> iteration.name  # doctest: +SKIP
    '11'
    >>> iteration.get_process_params()  # doctest: +SKIP
    {'dt': 0.75,
     'highpass': 0.01,
     'lowpass': 0.02,
     'npts': 500}

    Use ``$ lasif shell`` to play around and figure out what the iteration
    objects can do.
    """
    data = np.ones(npts * 2, dtype=np.float64)
    data[:npts] = 0.0

    # Use dummy trace for simple signal processing.
    tr = obspy.Trace(data=data)
    tr.stats.delta = delta

    # Use two band pass filters to get some time shift and band limit the data.
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(0.05, type="cosine")
    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=3,
              zerophase=False)
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(0.05, type="cosine")
    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=3,
              zerophase=False)

    # Final cut. It's really important to make sure that the first sample in
    # the stf is actually zero!
    tr.data = tr.data[npts:]

    return tr.data
