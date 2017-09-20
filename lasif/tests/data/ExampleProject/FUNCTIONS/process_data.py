#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project specific function processing observed data.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
from scipy import signal
from lasif import LASIFError


def processing_function(st, inv, processing_params, event):  # NOQA
    """
    Function to perform the actual preprocessing for one individual seismogram.
    This is part of the project so it can change depending on the project.

    Please keep in mind that you will have to manually update this file to a
    new version if LASIF is ever updated.

    You can do whatever you want in this function as long as the function
    signature is honored. The file is read from ``"input_filename"`` and
    written to ``"output_filename"``.

    One goal of this function is to make sure that the data is available at the
    same time steps as the synthetics. The first time sample of the synthetics
    will always be the origin time of the event.

    Furthermore the data has to be converted to m/s.

    :param processing_info: A dictionary containing information about the
        file to be processed. It will have the following structure.
    :type processing_info: dict

    .. code-block:: python

        {'event_information': {
            'depth_in_km': 22.0,
            'event_name': 'GCMT_event_VANCOUVER_ISLAND...',
            'filename': '/.../GCMT_event_VANCOUVER_ISLAND....xml',
            'latitude': 49.53,
            'longitude': -126.89,
            'm_pp': 2.22e+18,
            'm_rp': -2.78e+18,
            'm_rr': -6.15e+17,
            'm_rt': 1.98e+17,
            'm_tp': 5.14e+18,
            'm_tt': -1.61e+18,
            'magnitude': 6.5,
            'magnitude_type': 'Mwc',
            'origin_time': UTCDateTime(2011, 9, 9, 19, 41, 34, 200000),
            'region': u'VANCOUVER ISLAND, CANADA REGION'},
         'input_filename': u'/.../raw/7D.FN01A..HHZ.mseed',
         'output_filename': u'/.../processed_.../7D.FN01A..HHZ.mseed',
         'process_params': {
            'dt': 0.75,
            'highpass': 0.007142857142857143,
            'lowpass': 0.0125,
            'npts': 2000},
         'station_coordinates': {
            'elevation_in_m': -54.0,
            'latitude': 46.882,
            'local_depth_in_m': None,
            'longitude': -124.3337},
         'station_filename': u'/.../STATIONS/RESP/RESP.7D.FN01A..HH*'}

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
    def zerophase_chebychev_lowpass_filter(trace, freqmax):
        """
        Custom Chebychev type two zerophase lowpass filter useful for
        decimation filtering.

        This filter is stable up to a reduction in frequency with a factor of
        10. If more reduction is desired, simply decimate in steps.

        Partly based on a filter in ObsPy.

        :param trace: The trace to be filtered.
        :param freqmax: The desired lowpass frequency.

        Will be replaced once ObsPy has a proper decimation filter.
        """
        # rp - maximum ripple of passband, rs - attenuation of stopband
        rp, rs, order = 1, 96, 1e99
        ws = freqmax / (trace.stats.sampling_rate * 0.5)  # stop band frequency
        wp = ws  # pass band frequency

        while True:
            if order <= 12:
                break
            wp *= 0.99
            order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)

        b, a = signal.cheby2(order, rs, wn, btype="low", analog=0, output="ba")

        # Apply twice to get rid of the phase distortion.
        trace.data = signal.filtfilt(b, a, trace.data)

    # =========================================================================
    # Gather basic information.
    # =========================================================================
    npts = processing_params["npts"]
    dt = processing_params["dt"]
    min_period = processing_params['highpass_period']
    max_period = processing_params['lowpass_period']

    starttime = event["origin_time"] + processing_params["salvus_start_time"]
    endtime = starttime + processing_params["dt"] * \
        (processing_params["npts"] - 1)
    duration = endtime - starttime

    f2 = 0.9 / max_period
    f3 = 1.1 / min_period
    # Recommendations from the SAC manual.
    f1 = 0.5 * f2
    f4 = 2.0 * f3
    pre_filt = (f1, f2, f3, f4)

    for tr in st:
        # Make sure the seismograms are long enough. If not, skip them.
        if starttime < tr.stats.starttime or endtime > tr.stats.endtime:

            msg = ("The seismogram does not cover the required time span.\n"
                   "Seismogram time span: %s - %s\n"
                   "Requested time span: %s - %s" % (
                       tr.stats.starttime, tr.stats.endtime,
                       starttime, endtime))
            raise LASIFError(msg)

        # Trim to reduce processing cost.
        tr.trim(starttime - 0.2 * duration, endtime + 0.2 * duration)

        # =====================================================================
        # Some basic checks on the data.
        # =====================================================================
        # Non-zero length
        if not len(tr):
            msg = "No data found in time window around the event." \
                  " File skipped."
            raise LASIFError(msg)

        # No nans or infinity values allowed.
        if not np.isfinite(tr.data).all():
            msg = "Data contains NaNs or Infs. File skipped"
            raise LASIFError(msg)

        # =====================================================================
        # Step 1: Decimation
        # Decimate with the factor closest to the sampling rate of the
        # synthetics.
        # The data is still oversampled by a large amount so there should be no
        # problems. This has to be done here so that the instrument correction
        # is reasonably fast even for input data with a large sampling rate.
        # =====================================================================
        while True:
            decimation_factor = int(dt / tr.stats.delta)
            # Decimate in steps for large sample rate reductions.
            if decimation_factor > 8:
                decimation_factor = 8
            if decimation_factor > 1:
                new_nyquist = tr.stats.sampling_rate / 2.0 / float(
                    decimation_factor)
                zerophase_chebychev_lowpass_filter(tr, new_nyquist)
                tr.decimate(factor=decimation_factor, no_filter=True)
            else:
                break

        # =====================================================================
        # Step 2: Detrend and taper.
        # =====================================================================
        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(max_percentage=0.05, type="hann")

        # =====================================================================
        # Step 3: Instrument correction
        # Correct seismograms to velocity in m/s.
        # ======================================================================
        try:
            tr.attach_response(inv)
            tr.remove_response(output="DISP", pre_filt=pre_filt,
                               zero_mean=False, taper=False)
        except Exception as e:
            station = tr.stats.network + "." + tr.stats.station + ".." + \
                tr.stats.channel
            msg = ("File  could not be corrected with the help of the "
                   "StationXML file '%s'. Due to: '%s'  Will be skipped.") % \
                  (station, e.__repr__()),
            raise LASIFError(msg)

        # =====================================================================
        # Step 4: Bandpass filtering
        # This has to be exactly the same filter as in the source time function
        # in the case of SES3D.
        # =====================================================================
        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(0.05, type="cosine")
        tr.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3,
                  zerophase=False)
        tr.detrend("linear")
        tr.detrend("demean")
        tr.taper(0.05, type="cosine")
        tr.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3,
                  zerophase=False)

        # =====================================================================
        # Step 5: Sinc interpolation
        # =====================================================================
        tr.data = np.require(tr.data, requirements="C")
        tr.interpolate(sampling_rate=1.0 / dt, method="lanczos",
                       starttime=starttime, window="blackman", a=12, npts=npts)

        # =====================================================================
        # Save processed data and clean up.
        # =====================================================================
        # Convert to single precision to save some space.
        tr.data = np.require(tr.data, dtype="float32", requirements="C")
        if hasattr(tr.stats, "mseed"):
            tr.stats.mseed.encoding = "FLOAT32"

    return st
