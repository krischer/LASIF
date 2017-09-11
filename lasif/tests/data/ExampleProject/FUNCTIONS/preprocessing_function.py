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
import obspy
from obspy.io.xseed import Parser
from scipy import signal
import warnings

from lasif import LASIFError


def preprocessing_function(processing_info, iteration):  # NOQA
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
    # Read seismograms and gather basic information.
    # =========================================================================
    starttime = processing_info["event_information"]["origin_time"]
    endtime = starttime + processing_info["process_params"]["dt"] * \
        (processing_info["process_params"]["npts"] - 1)
    duration = endtime - starttime

    st = obspy.read(processing_info["input_filename"])

    if len(st) != 1:
        warnings.warn("The file '%s' has %i traces and not 1. "
                      "Skip all but the first" % (
                          processing_info["input_filename"], len(st)))
    tr = st[0]

    # Make sure the seismograms are long enough. If not, skip them.
    if starttime < tr.stats.starttime or endtime > tr.stats.endtime:

        msg = ("The seismogram does not cover the required time span.\n"
               "Seismogram time span: %s - %s\n"
               "Requested time span: %s - %s" % (
                   tr.stats.starttime, tr.stats.endtime, starttime, endtime))
        raise LASIFError(msg)

    # Trim to reduce processing cost.
    # starttime is the origin time of the event
    # endtime is the origin time plus the length of the synthetics
    tr.trim(starttime - 0.2 * duration, endtime + 0.2 * duration)

    # =========================================================================
    # Some basic checks on the data.
    # =========================================================================
    # Non-zero length
    if not len(tr):
        msg = "No data found in time window around the event. File skipped."
        raise LASIFError(msg)

    # No nans or infinity values allowed.
    if not np.isfinite(tr.data).all():
        msg = "Data contains NaNs or Infs. File skipped"
        raise LASIFError(msg)

    # =========================================================================
    # Step 1: Decimation
    # Decimate with the factor closest to the sampling rate of the synthetics.
    # The data is still oversampled by a large amount so there should be no
    # problems. This has to be done here so that the instrument correction is
    # reasonably fast even for input data with a large sampling rate.
    # =========================================================================
    while True:
        decimation_factor = int(processing_info["process_params"]["dt"] /
                                tr.stats.delta)
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

    # =========================================================================
    # Step 2: Detrend and taper.
    # =========================================================================
    tr.detrend("linear")
    tr.detrend("demean")
    tr.taper(max_percentage=0.05, type="hann")

    # =========================================================================
    # Step 3: Instrument correction
    # Correct seismograms to velocity in m/s.
    # =========================================================================
    output_units = "VEL"
    station_file = processing_info["station_filename"]

    # check if the station file actually exists ==============================
    if not processing_info["station_filename"]:
        msg = "No station file found for the relevant time span. File skipped"
        raise LASIFError(msg)

    # This is really necessary as other filters are just not sharp enough
    # and lots of energy from other frequency bands leaks into the frequency
    # band of interest
    freqmin = processing_info["process_params"]["highpass"]
    freqmax = processing_info["process_params"]["lowpass"]

    f2 = 0.9 * freqmin
    f3 = 1.1 * freqmax
    # Recommendations from the SAC manual.
    f1 = 0.5 * f2
    f4 = 2.0 * f3
    pre_filt = (f1, f2, f3, f4)

    # processing for seed files ==============================================
    if "/SEED/" in station_file:
        # XXX: Check if this is m/s. In all cases encountered so far it
        # always is, but SEED is in theory also able to specify corrections
        # to other units...
        parser = Parser(station_file)
        try:
            # The simulate might fail but might still modify the data. The
            # backup is needed for the backup plan to only correct using
            # poles and zeros.
            backup_tr = tr.copy()
            try:
                tr.simulate(seedresp={"filename": parser,
                                      "units": output_units,
                                      "date": tr.stats.starttime},
                            pre_filt=pre_filt, zero_mean=False, taper=False)
            except ValueError:
                warnings.warn("Evalresp failed, will only use the Poles and "
                              "Zeros stage")
                tr = backup_tr
                paz = parser.get_paz(tr.id, tr.stats.starttime)
                if paz["sensitivity"] == 0:
                    warnings.warn("Sensitivity is 0 in SEED file and will "
                                  "not be taken into account!")
                    tr.simulate(paz_remove=paz, remove_sensitivity=False,
                                pre_filt=pre_filt, zero_mean=False,
                                taper=False)
                else:
                    tr.simulate(paz_remove=paz, pre_filt=pre_filt,
                                zero_mean=False, taper=False)
        except Exception as e:
            msg = ("File  could not be corrected with the help of the "
                   "SEED file '%s'. Will be skipped due to: %s") \
                % (processing_info["station_filename"], str(e))
            raise LASIFError(msg)
    # processing with RESP files =============================================
    elif "/RESP/" in station_file:
        try:
            tr.simulate(seedresp={"filename": station_file,
                                  "units": output_units,
                                  "date": tr.stats.starttime},
                        pre_filt=pre_filt, zero_mean=False, taper=False)
        except ValueError as e:
            msg = ("File  could not be corrected with the help of the "
                   "RESP file '%s'. Will be skipped. Due to: %s") \
                % (processing_info["station_filename"], str(e))
            raise LASIFError(msg)
    elif "/StationXML/" in station_file:
        try:
            inv = obspy.read_inventory(station_file, format="stationxml")
        except Exception as e:
            msg = ("Could not open StationXML file '%s'. Due to: %s. Will be "
                   "skipped." % (station_file, str(e)))
            raise LASIFError(msg)
        tr.attach_response(inv)
        try:
            tr.remove_response(output=output_units, pre_filt=pre_filt,
                               zero_mean=False, taper=False)
        except Exception as e:
            msg = ("File  could not be corrected with the help of the "
                   "StationXML file '%s'. Due to: '%s'  Will be skipped.") \
                % (processing_info["station_filename"], e.__repr__()),
            raise LASIFError(msg)
    else:
        raise NotImplementedError

    # =========================================================================
    # Step 4: Bandpass filtering
    # This has to be exactly the same filter as in the source time function
    # in the case of SES3D.
    # =========================================================================
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

    # =========================================================================
    # Step 5: Sinc interpolation
    # =========================================================================
    # Make sure that the data array is at least as long as the
    # synthetics array.
    tr.interpolate(
        sampling_rate=1.0 / processing_info["process_params"]["dt"],
        method="lanczos", starttime=starttime, window="blackman", a=12,
        npts=processing_info["process_params"]["npts"])

    # =========================================================================
    # Save processed data and clean up.
    # =========================================================================
    # Convert to single precision to save some space.
    tr.data = np.require(tr.data, dtype="float32", requirements="C")
    if hasattr(tr.stats, "mseed"):
        tr.stats.mseed.encoding = "FLOAT32"

    tr.write(processing_info["output_filename"], format=tr.stats._format)
