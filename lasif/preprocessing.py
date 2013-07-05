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
import colorama
import multiprocessing
import numpy as np
import obspy
from obspy.xseed import Parser
from Queue import Full as QueueFull
from Queue import Empty as QueueEmpty
from scipy.interpolate import interp1d
import sys
import time
import warnings


# File wide lock for reading/writing MiniSEED files.
lock = multiprocessing.Lock()


def preprocess_file(file_info):
    """
    Function to perform the actual preprocessing.

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
        * file_number: The number of the file being processed. Useful for
            progress messages.

    Please remember to not touch the lock/mutexes, otherwise it will break with
    current versions of ObsPy. It is probably a good idea to have serial I/O in
    any case, at least with normal HDD. SSD might be a different issue.
    """
    sys.stdout.write("Processing file %i: " % file_info["file_number"])
    sys.stdout.write("%s \n" % file_info["data_path"])
    sys.stdout.flush()
    starttime = file_info["origin_time"]
    endtime = starttime + file_info["dt"] * (file_info["npts"] - 1)
    duration = endtime - starttime

    # The lock is necessary for MiniSEED files. This is a limitation of the
    # current version of ObsPy and will hopefully be resolved soon!
    # Do not remove it!
    lock.acquire()
    st = obspy.read(file_info["data_path"])
    lock.release()

    if len(st) != 1:
        msg = ("Warning: File '%s' has %i traces and not 1. "
            "Will be skipped") % (file_info["data_path"], len(st))
        warnings.warn(msg)
    tr = st[0]

    #==========================================================================
    # Detrend and taper before filtering and response removal
    #==========================================================================

    # Trim with a short buffer in an attempt to avoid boundary effects.
    # starttime is the origin time of the event
    # endtime is the origin time plus the length of the synthetics
    tr.trim(starttime - 0.05 * duration, endtime + 0.05 * duration)

    if len(tr) == 0:
        msg = ("Warning: After trimming the file '%s' to "
            "a time window around the event, no more data is "
            "left. The reference time is the one given in the "
            "QuakeML file. Make sure it is correct and that "
            "the waveform data actually contains data in that "
            "time span.") % file_info["data_path"]
        warnings.warn(msg)

    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper()

    #==========================================================================
    # Instrument correction
    #==========================================================================

    # Decimate in case there is a large difference between synthetic
    # sampling rate and sampling_rate of the data to accelerate the
    # process..
    # XXX: Ugly filter, change!
    if tr.stats.sampling_rate > (6 * 1.0 / file_info["dt"]):
        new_nyquist = tr.stats.sampling_rate / 2.0 / 5.0
        tr.filter("lowpass", freq=new_nyquist, corners=4, zerophase=True)
        tr.decimate(factor=5, no_filter=True)

    # Remove instrument response to produce velocity seismograms
    station_file = file_info["station_filename"]
    if "/SEED/" in station_file:
        paz = Parser(station_file).getPAZ(tr.id, tr.stats.starttime)
        try:
            tr.simulate(paz_remove=paz)
        except ValueError:
            msg = "Warning: Response of '%s' could not be removed. Skipped."\
                % file_info["data_path"]
            warnings.warn(msg)
            return True
    elif "/RESP/" in station_file:
        try:
            tr.simulate(seedresp={"filename": station_file, "units": "VEL",
                "date": tr.stats.starttime})
        except ValueError:
            msg = "Warning: Response of '%s' could not be removed. Skipped."\
                % file_info["data_path"]
            warnings.warn(msg)
            return True
    else:
        raise NotImplementedError

    #==========================================================================
    # Bandpass and interpolation
    #==========================================================================

    # Make sure that the data array is at least as long as the
    # synthetics array. Also add some buffer sample for the
    # spline interpolation to work in any case.
    buf = file_info["dt"] * 5
    if starttime < (tr.stats.starttime + buf):
        tr.trim(starttime=starttime - buf, pad=True, fill_value=0.0)
    if endtime > (tr.stats.endtime - buf):
        tr.trim(endtime=endtime + buf, pad=True, fill_value=0.0)

    # This is exactly the same filter as in the source time function. Should
    # eventually be configurable.
    tr.filter("lowpass", freq=file_info["lowpass"], corners=5, zerophase=False)
    tr.filter("highpass", freq=file_info["highpass"], corners=2,
        zerophase=False)

    # Decimation.
    factor = int(round(file_info["dt"] / tr.stats.delta))
    try:
        tr.decimate(factor, no_filter=True)
    except ValueError:
        msg = "Warning: File '%s' could not be decimated. Skipped." % \
            file_info["data_path"]
        warnings.warn(msg)
        return True

    # Interpolation.
    new_time_array = np.linspace(starttime.timestamp, endtime.timestamp,
        file_info["npts"])
    old_time_array = np.linspace(tr.stats.starttime.timestamp,
        tr.stats.endtime.timestamp, tr.stats.npts)

    tr.data = interp1d(old_time_array, tr.data, kind=1)(new_time_array)
    tr.stats.starttime = starttime
    tr.stats.delta = file_info["dt"]

    # Convert to single precision.
    tr.data = np.require(tr.data, dtype="float32", requirements="C")
    if hasattr(tr.stats, "mseed"):
        tr.stats.mseed.encoding = "FLOAT32"

    # The lock is necessary for MiniSEED files. This is a limitation of the
    # current version of ObsPy and will hopefully be resolved soon!
    # Do not remove it!
    lock.acquire()
    tr.write(file_info["processed_data_path"], format=tr.stats._format)
    lock.release()

    return True


def worker(receiving_queue, sending_queue):
    # Use None as the poison pill to stop the worker.
    for func, args, counter in iter(receiving_queue.get, None):
        args["file_number"] = counter
        result = func(args)
        sending_queue.put(result)


def pool_imap_unordered(function, iterable, processes):
    """
    Custom unordered map implementation. The advantage of this is that it, in
    contrast to the default Pool.map/imap implementations, does not consume the
    whole iterable upfront but only when it needs them. This enables infinitely
    long input queues. This might actually become relevant for LASIF if enough
    data is present.
    """
    # Creating the queues for sending and receiving items from the iterable.
    sending_queue = multiprocessing.Queue(processes)
    receiving_queue = multiprocessing.Queue()

    # Start the worker processes.
    for rpt in xrange(processes):
        multiprocessing.Process(target=worker, args=(sending_queue,
            receiving_queue)).start()

    # Iterate over the iterable and communicate with the worker process.
    send_len = 0
    recv_len = 0

    try:
        value = iterable.next()
        while True:
            time.sleep(0.1)
            try:
                sending_queue.put((function, value, send_len + 1), True, 0.2)
                send_len += 1
                value = iterable.next()
            except QueueFull:
                while True:
                    try:
                        result = receiving_queue.get(False)
                        recv_len += 1
                        yield result
                    except QueueEmpty:
                        break
    except StopIteration:
        pass

    # Collect all remaining results.
    while recv_len < send_len:
        result = receiving_queue.get(True, 10.0)
        recv_len += 1
        yield result

    # Terminate the worker processes but passing the poison pill.
    for rpt in xrange(processes):
        sending_queue.put(None)


def launch_processing(data_generator):
    # Use twice as many processes as core. The whole operations does a lot of
    # I/O thus more time is available for calculations.
    processes = 2 * multiprocessing.cpu_count()

    print ("%sLaunching preprocessing using %i processes...%s\n"
        "This might take a while. Press Ctrl + C to cancel.\n") % (
        colorama.Fore.GREEN, processes, colorama.Style.RESET_ALL)

    # Give the user some time to read the message.
    time.sleep(7.5)

    file_count = 0
    for i in pool_imap_unordered(preprocess_file, data_generator, processes):
        file_count += 1

    return file_count
