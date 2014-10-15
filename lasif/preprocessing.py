#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functionality for data preprocessing.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import time
import warnings

from lasif import LASIFError
from lasif.tools.colored_logger import ColoredLogger
from lasif.tools.parallel_helpers import parallel_map


def preprocess_file(processing_info):
    """
    Function to perform the actual preprocessing for one individual seismogram.

    One goal of this function is to make sure that the data is available at the
    same time steps as the synthetics. The first time sample of the synthetics
    will always be the origin time of the event.

    Furthermore the data has to be converted to m/s.

    :param processing_info: A dictionary containing information about the
        file to be processed. It will have the following structure.

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
            'npts': 2000,
            'stf': 'Filtered Heaviside'},
         'station_coordinates': {
            'elevation_in_m': -54.0,
            'latitude': 46.882,
            'local_depth_in_m': None,
            'longitude': -124.3337},
         'station_filename': u'/.../STATIONS/RESP/RESP.7D.FN01A..HH*'}

    """
    import numpy as np
    import obspy
    from obspy.xseed import Parser
    from scipy import signal
    from scipy.interpolate import interp1d

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

    # Trim with a short buffer in an attempt to avoid boundary effects.
    # starttime is the origin time of the event
    # endtime is the origin time plus the length of the synthetics
    tr.trim(starttime - 0.05 * duration, endtime + 0.05 * duration)

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
    # Step 1: Detrend and taper.
    # =========================================================================
    tr.detrend("linear")
    tr.taper(0.05, type="hann")

    # =========================================================================
    # Step 2: Decimation
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
    # Step 3: Instrument correction
    # Correct seismograms to velocity in m/s.
    # =========================================================================
    station_file = processing_info["station_filename"]

    print station_file

    # check if the station file actually exists ==============================
    if not processing_info["station_filename"]:
        msg = "No station file found for the relevant time span. File skipped"
        raise LASIFError(msg)

    # processing for seed files ==============================================
    if "/SEED/" in station_file:
        # XXX: Check if this is m/s. In all cases encountered so far it
        # always is, but SEED is in theory also able to specify corrections
        # to other units...
        paz = Parser(station_file).getPAZ(tr.id, tr.stats.starttime)
        try:
            tr.simulate(paz_remove=paz)
        except ValueError:
            msg = ("File  could not be corrected with the help of the "
                   "SEED file '%s'. Will be skipped.") \
                % processing_info["station_filename"],
            raise LASIFError(msg)

    # processing with RESP files =============================================
    elif "/RESP/" in station_file:
        try:
            tr.simulate(seedresp={"filename": station_file, "units": "VEL",
                                  "date": tr.stats.starttime})
        except ValueError:
            msg = ("File  could not be corrected with the help of the "
                   "RESP file '%s'. Will be skipped.") \
                % processing_info["station_filename"],
            raise LASIFError(msg)
    elif "/StationXML/" in station_file:
        try:
            inv = obspy.read_inventory(station_file, format="stationxml")
        except Exception as e:
            msg = ("Could not open StationXML file '%s'. Due to %s. Will be "
                   "skipped." % (station_file, str(e)))
            raise LASIFError(msg)
        tr.attach_response(inv)
        try:
            tr.remove_response()
        except:
            msg = ("File  could not be corrected with the help of the "
                   "StationXML file '%s'. Will be skipped.") \
                  % processing_info["station_filename"],
            raise LASIFError(msg)
    else:
        raise NotImplementedError



    # =========================================================================
    # Step 4: Interpolation
    # =========================================================================
    # Apply one more taper to avoid high frequency contributions from sudden
    # steps at the beginning/end if padded with zeros.
    tr.taper(0.05, type="hann")
    # Make sure that the data array is at least as long as the
    # synthetics array. Also add some buffer sample for the
    # spline interpolation to work in any case.
    buf = processing_info["process_params"]["dt"] * 5
    if starttime < (tr.stats.starttime + buf):
        tr.trim(starttime=starttime - buf, pad=True, fill_value=0.0)
    if endtime > (tr.stats.endtime - buf):
        tr.trim(endtime=endtime + buf, pad=True, fill_value=0.0)

    # Actual interpolation. Currently a linear interpolation is used.
    new_time_array = np.linspace(starttime.timestamp, endtime.timestamp,
                                 processing_info["process_params"]["npts"])
    old_time_array = np.linspace(tr.stats.starttime.timestamp,
                                 tr.stats.endtime.timestamp, tr.stats.npts)
    tr.data = interp1d(old_time_array, tr.data, kind=1)(new_time_array)
    tr.stats.starttime = starttime
    tr.stats.delta = processing_info["process_params"]["dt"]

    # =========================================================================
    # Step 5: Bandpass filtering
    # This has to be exactly the same filter as in the source time function.
    # Should eventually be configurable.
    # =========================================================================
    tr.filter("lowpass", freq=processing_info["process_params"]["lowpass"],
              corners=5, zerophase=False)
    tr.filter("highpass", freq=processing_info["process_params"]["highpass"],
              corners=2, zerophase=False)

    # =========================================================================
    # Save processed data and clean up.
    # =========================================================================
    # Convert to single precision for saving.
    tr.data = np.require(tr.data, dtype="float32", requirements="C")
    if hasattr(tr.stats, "mseed"):
        tr.stats.mseed.encoding = "FLOAT32"

    tr.write(processing_info["output_filename"], format=tr.stats._format)


def launch_processing(data_generator, log_filename=None, waiting_time=4.0,
                      process_params=None):
    """
    Launch the parallel processing.

    :param data_generator: A generator yielding file information as required.
    :param log_filename: If given, a log will be written to that file.
    :param waiting_time: The time spent sleeping after the initial message has
        been printed. Useful if the user should be given the chance to cancel
        the processing.
    :param process_params: If given, the processing parameters will be written
        to the logfile.
    """
    logger = ColoredLogger(log_filename=log_filename)

    logger.info("Launching preprocessing using all processes...\n"
                "This might take a while. Press Ctrl + C to cancel.\n")

    # Give the user some time to read the message.
    time.sleep(waiting_time)
    results = parallel_map(preprocess_file,
                           ({"processing_info": i} for i in data_generator),
                           verbose=50, pre_dispatch="all")

    # Keep track of all files.
    successful_file_count = 0
    warning_file_count = 0
    failed_file_count = 0
    total_file_count = len(results)

    for result in results:
        if result.exception is not None:
            filename = result.func_args["processing_info"]["input_filename"]
            msg = "Exception processing file '%s'. %s\n%s" % (
                filename, result.exception, result.traceback)
            logger.error(msg)
            failed_file_count += 1
        elif result.warnings:
            warning_file_count += 1
        else:
            successful_file_count += 1

    return {
        "failed_file_count": failed_file_count,
        "warning_file_count": warning_file_count,
        "total_file_count": total_file_count,
        "successful_file_count": successful_file_count}
