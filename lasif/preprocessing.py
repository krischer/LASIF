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
import numpy as np
from numpy.distutils import system_info
import obspy
from obspy.xseed import Parser
import os
import platform
import Queue
from scipy.interpolate import interp1d
from scipy import signal
import sys
import time
import traceback
import warnings

from lasif.project import LASIFException
from lasif.tools.colored_logger import ColoredLogger


# The veclib/accelerate framework on OSX is not compatible with
# multiprocessing. Thus only one core will be used on those. The only
# proper solution to this is to install numpy/scipy with OpenBLAS and not
# the Accelerate framework.
multiproc = True
if platform.system().lower() == "darwin":
    blas_info = system_info.blas_opt_info().get_info()
    if "extra_link_args" in blas_info:
        if "Accelerate" in str(blas_info["extra_link_args"]):
            msg = ("The numpy installation on this computer is linked "
                   "against the Accelerate/veclib framework of OSX. "
                   "This is incompatible with multiprocessing and it "
                   "appears that there are no intentions to fix it. "
                   "For this reason only one process will be used for the "
                   "processing. The only proper fix is to link numpy "
                   "against OpenBLAS. Google helps.")
            warnings.warn(msg)
            multiproc = False

if multiproc:
    import multiprocessing
    parallel_class = multiprocessing.Process
    queue_class = multiprocessing.Queue
    process_count = multiprocessing.cpu_count()
    # Lock used to synchronize I/O.
    lock = multiprocessing.Lock()
else:
    import threading
    parallel_class = threading.Thread
    queue_class = Queue.Queue
    process_count = 1
    # Lock used to synchronize I/O.
    lock = threading.Lock()


def zerophase_chebychev_lowpass_filter(trace, freqmax):
    """
    Custom Chebychev type two zerophase lowpass filter useful for decimation
    filtering.

    This filter is stable up to a reduction in frequency with a factor of 10.
    If more reduction is desired, simply decimate in steps.

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


def preprocess_file(file_info):
    """
    Function to perform the actual preprocessing for one individual seismogram.

    One goal of this function is to make sure that the data is available at the
    same time steps as the synthetics. The first time sample of the synthetics
    will always be the origin time of the event.

    Furthermore the data has to be converted to m/s.

    :param file_info: A dictionary containing information about the file to
        be processed.

    file_info is dictionary containing the following keys:
        * data_path: Path to the raw data. This has to be read.
        * processed_data_path: The path where the processed file has to be
            saved at.
        * origin_time: The origin time of the event for the file.
        * npts: The number of samples of the synthetic waveform. The data
            should be interpolated to have the same amount of samples.
        * dt: The time increment of the synthetics.
        * station_filename: The filename of the station file for the waveform.
            Can either be a RESP file or a dataless SEED file.
        * highpass: The highpass frequency of the source time function for the
            synthetics.
        * lowpass: The lowpass frequency of the source time function for the
            synthetics.
    """
    warning_messages = []

    #==========================================================================
    # Read seismograms and gather basic information.
    #==========================================================================
    starttime = file_info["origin_time"]
    endtime = starttime + file_info["dt"] * (file_info["npts"] - 1)
    duration = endtime - starttime

    with lock:
        st = obspy.read(file_info["data_path"])

    if len(st) != 1:
        warning_messages.append("The file has %i traces and not 1. "
                                "Skip all but the first" % len(st))
    tr = st[0]

    # Trim with a short buffer in an attempt to avoid boundary effects.
    # starttime is the origin time of the event
    # endtime is the origin time plus the length of the synthetics
    tr.trim(starttime - 0.05 * duration, endtime + 0.05 * duration)

    #==========================================================================
    # Some basic checks on the data.
    #==========================================================================
    # Non-zero length
    if not len(tr):
        msg = "No data found in time window around the event. File skipped."
        raise LASIFException(msg)

    # No nans or infinity values allowed.
    if not np.isfinite(tr.data).all():
        msg = "Data contains NaNs or Infs. File skipped"
        raise LASIFException(msg)

    #==========================================================================
    # Step 1: Detrend and taper.
    #==========================================================================
    sys.stdout.flush()
    tr.detrend("linear")
    tr.taper(0.05, type="hann")

    sys.stdout.flush()

    #==========================================================================
    # Step 2: Decimation
    # Decimate with the factor closest to the sampling rate of the synthetics.
    # The data is still oversampled by a large amount so there should be no
    # problems. This has to be done here so that the instrument correction is
    # reasonably fast even for input data with a large sampling rate.
    #==========================================================================
    while True:
        decimation_factor = int(file_info["dt"] / tr.stats.delta)
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

    #==========================================================================
    # Step 3: Instrument correction
    # Correct seismograms to velocity in m/s.
    #==========================================================================
    station_file = file_info["station_filename"]

    #- check if the station file actually exists ==============================
    if not file_info["station_filename"]:
        msg = "No station file found for the relevant time span. File skipped"
        raise LASIFException(msg)

    #- processing for seed files ==============================================
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
                % file_info["station_filename"],
            raise LASIFException(msg)

    #- processing with RESP files =============================================
    elif "/RESP/" in station_file:
        try:
            tr.simulate(seedresp={"filename": station_file, "units": "VEL",
                                  "date": tr.stats.starttime})
        except ValueError:
            msg = ("File  could not be corrected with the help of the "
                   "RESP file '%s'. Will be skipped.") \
                % file_info["station_filename"],
            raise LASIFException(msg)
    else:
        raise NotImplementedError

    #==========================================================================
    # Step 4: Interpolation
    #==========================================================================
    # Apply one more taper to avoid high frequency contributions from sudden
    # steps at the beginning/end if padded with zeros.
    tr.taper(0.05, type="hann")
    # Make sure that the data array is at least as long as the
    # synthetics array. Also add some buffer sample for the
    # spline interpolation to work in any case.
    buf = file_info["dt"] * 5
    if starttime < (tr.stats.starttime + buf):
        tr.trim(starttime=starttime - buf, pad=True, fill_value=0.0)
    if endtime > (tr.stats.endtime - buf):
        tr.trim(endtime=endtime + buf, pad=True, fill_value=0.0)

    # Actual interpolation. Currently a linear interpolation is used.
    new_time_array = np.linspace(starttime.timestamp, endtime.timestamp,
                                 file_info["npts"])
    old_time_array = np.linspace(tr.stats.starttime.timestamp,
                                 tr.stats.endtime.timestamp, tr.stats.npts)
    tr.data = interp1d(old_time_array, tr.data, kind=1)(new_time_array)
    tr.stats.starttime = starttime
    tr.stats.delta = file_info["dt"]

    #==========================================================================
    # Step 5: Bandpass filtering
    # This has to be exactly the same filter as in the source time function.
    # Should eventually be configurable.
    #==========================================================================
    tr.filter("lowpass", freq=file_info["lowpass"], corners=5, zerophase=False)
    tr.filter("highpass", freq=file_info["highpass"], corners=2,
              zerophase=False)

    #==========================================================================
    # Save processed data and clean up.
    #==========================================================================

    # Convert to single precision for saving.
    tr.data = np.require(tr.data, dtype="float32", requirements="C")
    if hasattr(tr.stats, "mseed"):
        tr.stats.mseed.encoding = "FLOAT32"

    with lock:
        tr.write(file_info["processed_data_path"], format=tr.stats._format)

    return {"warnings": warning_messages}


def worker(input_queue, output_queue):
    """
    Queue for each worker.

    :param output_queue: The queue where the jobs are stored.
    :param input_queue: The quere where the results are stored.
    """
    # Use "STOP" as the poison pill to stop the worker.
    for func, args in iter(input_queue.get, "STOP"):

        result = {
            "exception": None,
            "filename": args["data_path"],
            "warnings": []
        }

        # Catch warnings to be able to log them.
        with warnings.catch_warnings(record=True) as w:
            # Function level try/except.
            try:
                res = func(args)
            except:
                exc_type, exc_class, tb = sys.exc_info()
                # Make sure it can be pickled to be passed between processes.
                tb = traceback.extract_tb(tb)
                result["exception"] = (exc_type, exc_class, tb)
            else:
                result["warnings"] = res["warnings"]

            if len(w):
                for warning in w:
                    # Catch one specific (benign) warning. This will usually
                    # come up when designing rather sharp IIR filters for low
                    # frequencies. The results are still ok. If it fails, it
                    # would be immediatly noticed.
                    if "badly conditioned filter coefficients (numerator" in \
                            str(warning.message).lower():
                        continue
                    result["warnings"].append(
                        "%s: %s" % (str(warning.category), warning.message))

        output_queue.put(result)


def pool_imap_unordered(function, iterable):
    """
    Custom unordered map implementation. The advantage of this is that it, in
    contrast to the default Pool.map/imap implementations, does not consume the
    whole iterable upfront but only when it needs them. This enables infinitely
    long input queues. This might actually become relevant for LASIF if enough
    data is present.

    :param function: The function to run for each item.
    :param iterable: The iterable yielding items.
    :param processes: The number of processes to launch.
    """
    input_queue = queue_class()
    output_queue = queue_class()

    # Start the worker processes.
    collected_workers = []
    for _ in xrange(process_count):
        collected_workers.append(
            parallel_class(target=worker, args=(input_queue, output_queue)))

    # Start all processes.
    for proc in collected_workers:
        proc.start()

    # Count all sent and received tasks.
    send_len = 0
    recv_len = 0

    while True:
        # Artificial delay to give some times for the queues to play catch-up.
        time.sleep(0.1)

        # First try to get as many ready results as possible.
        try:
            result = output_queue.get_nowait()
            recv_len += 1
        # If that fails create new jobs.
        except Queue.Empty:
            # Get new values until there are no left. Then break.
            try:
                value = iterable.next()
            except StopIteration:
                break
            while True:
                try:
                    # Create a new job with the value.
                    input_queue.put((function, value), timeout=25)
                    send_len += 1
                except Queue.Full:
                    try:
                        result = output_queue.get(timeout=10)
                        recv_len += 1
                    except Queue.Empty:
                        continue
                    else:
                        yield result
                else:
                    break

        else:
            yield result

    # Collect all remaining results. Use a long timeout to catch the final
    # files.
    while recv_len < send_len:
        # Artificial delay to give some times for the queues to play
        # catch-up.
        time.sleep(0.1)
        try:
            result = output_queue.get(True, 25.0)
            sys.stdout.flush()
        except Queue.Empty:
            result = {
                "exception": None,
                "filename": "Unknown",
                "warnings": []
            }
            # Create an exception that can be passed to the parent process.
            try:
                msg = "Processing did not finish (deadlock?)"
                raise LASIFException(msg)
            except:
                exc_type, exc_class, tb = sys.exc_info()
                # Make sure it can be pickled to be passed between processes.
                tb = traceback.extract_tb(tb)
                result["exception"] = (exc_type, exc_class, tb)
        finally:
            recv_len += 1
        yield result

    # Terminate the worker processes by passing the poison pill.
    for _ in xrange(process_count):
        input_queue.put("STOP")

    time.sleep(0.1)

    for proc in collected_workers:
        proc.join()

    # Force a shutdown if one or more processes got hung up somewhere.
    for proc in collected_workers:
        if proc.is_alive():
            try:
                proc.terminate()
            except:
                pass
            proc.join()


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

    logger.info("Launching preprocessing using %i processes...\n"
                "This might take a while. Press Ctrl + C to cancel.\n"
                % process_count)

    if process_params:
        import pprint
        logger.info("Used parameters: \n%s" % pprint.pformat(process_params))

    # Give the user some time to read the message.
    time.sleep(waiting_time)

    # Keep track of all files.
    successful_file_count = 0
    warning_file_count = 0
    failed_file_count = 0
    total_file_count = 0

    # Result is a dictionary with 3 keys:
    #   * filename: The name of the current file.
    #   * exception: None or an actual exception as returned by
    #                sys.exc_info()
    #   * warnings: Warnings as a list of strings.
    for i, result in enumerate(pool_imap_unordered(
            function=preprocess_file, iterable=data_generator)):
        warnings.simplefilter("always")
        total_file_count += 1
        msg = "Processed file %i: '%s'" % (
            i, os.path.relpath(result["filename"]))
        logger.info(msg)
        if result["exception"] is not None:
            logger.exception(result["exception"][1])
            failed_file_count += 1
        elif result["warnings"]:
            for warning in result["warnings"]:
                logger.warning(warning)
            warning_file_count += 1
        else:
            successful_file_count += 1

    return {
        "failed_file_count": failed_file_count,
        "warning_file_count": warning_file_count,
        "total_file_count": total_file_count,
        "successful_file_count": successful_file_count}
