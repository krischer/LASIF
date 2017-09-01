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
import numpy as np
from obspy.signal.invsim import cosine_sac_taper

def process_synthetics(st, event):  # NOQA
    """
    This function is called after a synthetic file has been read.

    Do whatever you need to do in here an return a potentially modified
    stream object. Make sure that anything you do works with the
    preprocessing function. LASIF expects data and synthetics to have
    exactly the same length before it can pick windows and calculate adjoint
    sources.

    Potential uses for this function are to shift synthetics in time if
    required or to apply some processing to them which LASIF by default does
    not do.

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

    ``event`` is a dictionary with information about the current event:

    >>> event  # doctest: +SKIP
    {'depth_in_km': 10.0,
     'event_name': 'NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20',
     'filename': '/path/to/file.xml',
     'latitude': 43.43,
     'longitude': 21.37,
     'm_pp': -4.449e+17,
     'm_rp': -5.705e+17,
     'm_rr': -1.864e+17,
     'm_rt': 2.52e+16,
     'm_tp': 4.049e+17,
     'm_tt': 6.313e+17,
     'magnitude': 5.9,
     'magnitude_type': 'Mwc',
     'origin_time': UTCDateTime(1980, 5, 18, 20, 3, 6, 900000),
     'region': 'NORTHWESTERN BALKAN REGION'}
    """
    # Currently a no-op.
    # This function will modify each waveform stream. It must
    # be called process() and it takes three arguments:
    #
    # * st: The obspy.Stream object with the waveforms.
    # * inv: The obspy.Inventory object with the metadata.
    # * tag: The name of the currently selected tag.

    # TODO add parameter to pass these:
    freqmax = 1 / 60.0
    freqmin = 1 / 100.0
    f2 = 0.9 * freqmin
    f3 = 1.1 * freqmax
    f1 = 0.5 * f2
    f4 = 2.0 * f3
    pre_filt = (f1, f2, f3, f4)

    # Detrend and taper.
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=0.05, type="hann")

    # Assuming displacement seismograms
    for tr in st:

        data = tr.data.astype(np.float64)
        orig_len = len(data)

        # smart calculation of nfft dodging large primes
        # noinspection PyProtectedMember
        from obspy.signal.util import _npts2nfft
        nfft = _npts2nfft(len(data))

        fy = 1.0 / (tr.stats.delta * 2.0)
        freqs = np.linspace(0, fy, nfft // 2 + 1)

        # Transform data to Frequency domain
        data = np.fft.rfft(data, n=nfft)
        # noinspection PyTypeChecker
        data *= cosine_sac_taper(freqs, flimit=pre_filt)
        data[-1] = abs(data[-1]) + 0.0j

        # transform data back into the time domain
        data = np.fft.irfft(data)[0:orig_len]

        # assign processed data and store processing informatio
        tr.data = data

        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(0.05, type="cosine")
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=3,
                  zerophase=True)
        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(0.05, type="cosine")
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=3, zerophase=True)

    for tr in st:
        tr.stats.starttime = 0
    return st

