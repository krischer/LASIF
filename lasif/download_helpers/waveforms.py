#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple function using bulk waveform download services.

Queries ArcLink, Neries and the IRIS webservices.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/licenses/gpl.html)
"""
import collections
import logging
from obspy import Stream
import obspy.arclink
import obspy.iris
import obspy.neries
import Queue
import threading
import time
import warnings


# Some networks have trouble with ArcLink - currently only Italian networks.
# These will be downloaded with the help of the obspy.neries webclient.
NON_WORKING_ARCLINK_NETWORKS = ["IV", "GU", "MN", "NI", "BA", "IX", "OB"]

# Disable the logging for suds as this produces ugly error messages. Something
# with the logging inside of suds is messed up.
logging.getLogger("suds").setLevel(logging.CRITICAL)


def download_waveforms(channels, starttime, endtime, minimumlength,
        save_trace_fct, arclink_user, logger=None):
    """
    :param save_trace_fct: Function taking a single obspy.Trace and storing it
        however the user sees fit.

    Returns the number of successful downloaded channels.
    """
    # dequeues are slightly more threadsafe then lists. Should not be necessary
    # due Python's GIL but better safe then sorry.
    failed_downloads = collections.deque()
    successful_downloads = collections.deque()

    minimum_duration = minimumlength * (endtime - starttime)

    # Download first from IRIS
    st = starttime.strftime("%Y-%jT%H:%M:%S")
    et = endtime.strftime("%Y-%jT%H:%M:%S")

    def to_bulkdatarequest(channels):
        for channel in channels:
            net, sta, loc, chan = channel.split(".")
            if loc == "":
                loc = "--"
            yield "%s %s %s %s %s %s" % (net, sta, loc, chan, st, et)

    bk = "\n".join(to_bulkdatarequest(channels))
    # Give a very high timeout to IRIS. It is usually pretty fast and here just
    # used for the bulkdataselect which very rarely might take a while for
    # obscure data sets.
    iris_client = obspy.iris.Client(timeout=180)
    if logger:
        logger.debug("Starting IRIS bulkdataselect download...")
    try:
        stream = iris_client.bulkdataselect(bk,
            minimumlength=(endtime - starttime) * 0.9)
    except Exception as e:
        if not e.message.lower().startswith("no waveform data available"):
            msg = "Problem downloading data from IRIS\n"
            err_msg = str(e)
            msg += "\t%s: %s" % (e.__class__.__name__, err_msg)
            if logger:
                logger.error(msg)
            else:
                warnings.warn(msg)
        else:
            if logger:
                logger.debug("No data available at IRIS for request.")
        # Dummy stream to be able to use the same logic later on.
        stream = Stream()

    for tr in stream:
        if not tr.stats.npts or (tr.stats.endtime - tr.stats.starttime) < \
                minimum_duration:
            continue
        save_trace_fct(tr)
        successful_downloads.append(tr.id)
        if logger:
            logger.info("Successfully downloaded %s from IRIS." % tr.id)

    # Now get a list of all failed downloads.
    for chan in channels:
        if chan in successful_downloads:
            continue
        failed_downloads.append(chan)

    # ArcLink Downloader
    class ArcLinkDownloadThread(threading.Thread):
        def __init__(self, queue):
            self.queue = queue
            super(ArcLinkDownloadThread, self).__init__()

        def run(self):
            while True:
                try:
                    channel = self.queue.get(False)
                except Queue.Empty:
                    break
                time.sleep(0.5)
                if logger:
                    logger.debug("Starting ArcLink download for %s..." %
                        channel)
                arc_client = obspy.arclink.Client(user=arclink_user,
                    timeout=60)
                try:
                    st = arc_client.getWaveform(*channel.split("."),
                        starttime=starttime, endtime=endtime, format="mseed")
                except Exception as e:
                    if e.message.lower() != "no data available":
                        msg = ("Failed to download %s from ArcLink because of "
                               "an error (%s: %s)") % (channel,
                               e.__class__.__name__, e.message)
                        if logger:
                            logger.error(msg)
                        else:
                            warnings.warn(msg)
                    else:
                        if logger:
                            logger.debug("No data available at ArcLink for %s."
                                % channel)
                    continue
                if not st:
                    if logger:
                        logger.debug("No data available at ArcLink for %s."
                            % channel)
                    continue
                if len(st) != 1 or (st[0].stats.endtime -
                        st[0].stats.starttime) < minimum_duration:
                    if logger:
                        if len(st) != 1:
                            msg = "More than one Trace found for channel %s." \
                                % channel
                            logger.debug(msg)
                        else:
                            msg = ("Trace %s is only %.2f seconds long (%.2f "
                               "seconds required)") % (channel,
                               st[0].stats.endtime - st[0].stats.starttime,
                               minimum_duration)
                            logger.warning(msg)
                    continue
                save_trace_fct(st[0])
                failed_downloads.remove(channel)
                successful_downloads.append(channel)
                if logger:
                    logger.info("Successfully downloaded %s from ArcLink." %
                        st[0].id)

    # Neries downloader
    class NeriesDownloadThread(threading.Thread):
        def __init__(self, queue):
            self.queue = queue
            super(NeriesDownloadThread, self).__init__()

        def run(self):
            while True:
                try:
                    channel = self.queue.get(False)
                except Queue.Empty:
                    break
                time.sleep(0.25)
                if logger:
                    logger.debug("Starting Neries download for %s..." %
                        channel)
                neries_client = obspy.neries.Client(user=arclink_user,
                    timeout=60)
                try:
                    st = neries_client.getWaveform(*channel.split("."),
                        starttime=starttime, endtime=endtime, format="mseed")
                except Exception as e:
                    msg = ("Failed to download %s from Neries because of "
                           "an error (%s: %s)") % (channel,
                           e.__class__.__name__, str(e.message))
                    if logger:
                        logger.error(msg)
                    else:
                        warnings.warn(msg)
                    continue
                # Neries returns an empty stream object if not data is
                # available.
                if not st:
                    if logger:
                        logger.debug("No data available at Neries for %s."
                            % channel)
                    continue
                if len(st) != 1 or (st[0].stats.endtime -
                        st[0].stats.starttime) < minimum_duration:
                    if logger:
                        if len(st) != 1:
                            msg = "More than one Trace found for channel %s." \
                                % channel
                            logger.debug(msg)
                        else:
                            msg = ("Trace %s is only %.2f seconds long (%.2f "
                               "seconds required. Will not be saved.)") % \
                               (channel, st[0].stats.endtime -
                               st[0].stats.starttime, minimum_duration)
                            logger.warning(msg)
                    continue
                save_trace_fct(st[0])
                failed_downloads.remove(channel)
                successful_downloads.append(channel)
                if logger:
                    logger.info("Successfully downloaded %s from Neries." %
                        st[0].id)

    # Create one large queue containing everything.
    arclink_queue = Queue.Queue()
    neries_queue = Queue.Queue()
    for channel in failed_downloads:
        # Skip certain networks that do not work with ArcLink for some reason.
        network = channel.split(".")[0]
        if network in NON_WORKING_ARCLINK_NETWORKS:
            neries_queue.put(channel)
            continue
        arclink_queue.put(channel)

    # Download from ArcLink.
    my_threads = []
    # Launch 20 threads at max. Might seem a large number but timeout is set to
    # 60 seconds and they start with 1 second between each other.
    thread_count = min(20, arclink_queue.qsize())
    for _i in xrange(thread_count):
        thread = ArcLinkDownloadThread(queue=arclink_queue)
        my_threads.append(thread)
        thread.start()
        time.sleep(1.0)

    for thread in my_threads:
        thread.join()

    # Download the missing data via Neries.
    my_threads = []
    # 20 parallel threads appear to be too much for neries. Restrict to 10 but
    # launch them faster.
    thread_count = min(10, neries_queue.qsize())
    for _i in xrange(thread_count):
        thread = NeriesDownloadThread(queue=neries_queue)
        my_threads.append(thread)
        thread.start()
        time.sleep(0.5)

    for thread in my_threads:
        thread.join()

    for chan in failed_downloads:
        msg = ("Failed to download %s due to no data available or previously "
            "raised error.") % chan
        if not logger:
            warnings.warn(msg)
            continue
        logger.warning(msg)
    return len(successful_downloads)
