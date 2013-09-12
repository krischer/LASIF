#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A function downloading dataless SEED and StationXML files.

Queries ArcLink and the IRIS webservices.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/licenses/gpl.html)
"""
from copy import deepcopy
import obspy.arclink
import obspy.iris
from obspy.xseed import Parser
import Queue
import StringIO
import threading
import time

from lasif import utils


def download_station_files(channels, save_station_fct, arclink_user,
                           logger=None):
    """
    Downloads station files.

    :param channels: A list of dictionaries, each describing a channel with the
        following keys: 'network', 'station', 'location', 'channel',
        'starttime', 'endtime'
    :param save_station_fct: A function called upon saving every file. Will be
        called with a StringIO instance containing the station file, e.g.
        save_station_fct(string_io_instance, network, station, location,
            channel, format)
        Format will be either "datalessSEED", "StationXML", or "RESP"
    """
    class ArcLinkDownloadThread(threading.Thread):
        def __init__(self, queue, successful_downloads, failed_downloads):
            self.queue = queue
            self.successful_downloads = successful_downloads
            self.failed_downloads = failed_downloads
            super(ArcLinkDownloadThread, self).__init__()

        def run(self):
            while True:
                try:
                    channel = self.queue.get(False)
                except Queue.Empty:
                    break
                network = channel["network"]
                station = channel["station"]
                location = channel["location"]
                chan = channel["channel"]
                starttime = channel["starttime"]
                endtime = channel["endtime"]
                channel_id = "%s.%s.%s.%s" % (network, station, location,
                                              chan)
                time.sleep(0.5)
                if logger:
                    logger.debug("Starting ArcLink download for %s..." %
                                 channel_id)
                # Telnet sometimes has issues...
                success = False
                for _i in xrange(3):
                    try:
                        arc_client = obspy.arclink.Client(user=arclink_user,
                                                          timeout=30)
                        success = True
                        break
                    except:
                        time.sleep(0.3)
                if success is False:
                    msg = (" A problem occured initializing ArcLink. Try "
                           "again later")
                    logger.error(msg)
                    failed_downloads.put(channel)
                    continue
                try:
                    memfile = StringIO.StringIO()
                    arc_client.saveResponse(
                        memfile, channel["network"],
                        channel["station"], channel["location"],
                        channel["channel"], starttime=channel["starttime"],
                        endtime=channel["endtime"], format="SEED")
                except Exception as e:
                    msg = "While downloading %s [%s to %s]: %s" % (
                        channel_id, channel["starttime"], channel["endtime"],
                        str(e))
                    logger.error(msg)
                    failed_downloads.put(channel)
                    continue
                memfile.seek(0, 0)
                # Read the file again and perform a sanity check.
                try:
                    parser = Parser(memfile)
                except:
                    msg = (
                        "Arclink did not return a valid dataless SEED file "
                        "for channel %s [%s-%s]") % (channel_id, starttime,
                                                     endtime)
                    logger.error(msg)
                    failed_downloads.put(channel)
                    continue
                if not utils.channel_in_parser(parser, channel_id, starttime,
                                               endtime):
                    msg = (
                        "Arclink returned a valid dataless SEED file "
                        "for channel %s [%s to %s], but it does not actually "
                        " contain data for the requested channel and time "
                        "frame.") % \
                        (channel_id, starttime, endtime)
                    logger.error(msg)
                    failed_downloads.put(channel)
                    continue
                memfile.seek(0, 0)
                save_station_fct(
                    memfile, channel["network"],
                    channel["station"], channel["location"],
                    channel["channel"], format="datalessSEED")
                successful_downloads.put(channel)
                if logger:
                    logger.info("Successfully downloaded dataless SEED for "
                                "channel %s.%s.%s.%s from ArcLink." % (
                                    channel["network"], channel["station"],
                                    channel["location"], channel["channel"]))

    class IRISDownloadThread(threading.Thread):
        def __init__(self, queue, successful_downloads):
            self.queue = queue
            self.successful_downloads = successful_downloads
            super(IRISDownloadThread, self).__init__()

        def run(self):
            while True:
                try:
                    channel = self.queue.get(False)
                except Queue.Empty:
                    break
                network = channel["network"]
                station = channel["station"]
                location = channel["location"]
                chan = channel["channel"]
                channel_id = "%s.%s.%s.%s" % (network, station, location,
                                              chan)
                time.sleep(0.5)
                if logger:
                    logger.debug("Starting IRIS download for %s..." %
                                 channel_id)
                client = obspy.iris.Client()
                try:
                    resp_data = client.resp(
                        channel["network"], channel["station"],
                        channel["location"], channel["channel"],
                        starttime=channel["starttime"],
                        endtime=channel["endtime"])
                except Exception as e:
                    msg = "While downloading %s from IRIS [%s to %s]: %s" % (
                        channel_id, channel["starttime"], channel["endtime"],
                        str(e))
                    logger.error(msg)
                    continue

                if not resp_data:
                    msg = ("While downloading %s from IRIS [%s to %s]: "
                           "No data returned") % (
                        channel_id, channel["starttime"],
                        channel["endtime"])
                    logger.error(msg)
                    continue

                memfile = StringIO.StringIO(resp_data)
                memfile.seek(0, 0)

                save_station_fct(memfile, channel["network"],
                                 channel["station"], channel["location"],
                                 channel["channel"], format="RESP")
                successful_downloads.put(channel)
                if logger:
                    logger.info(
                        "Successfully downloaded RESP file for "
                        "channel %s.%s.%s.%s from IRIS." % (
                            channel["network"], channel["station"],
                            channel["location"], channel["channel"]))

    # Attempt to group the ArcLink station downloads because it is nicer to
    # have one SEED file per station instead of per channel.
    a_channels = [{
        "network": _i["network"],
        "station": _i["station"],
        "location": _i["location"],
        "channel": _i["channel"][:2] + "*",
        "starttime": _i["starttime"],
        "endtime": _i["endtime"]} for _i in channels]
    arclink_channels = []
    for channel in a_channels:
        if channel in arclink_channels:
            continue
        arclink_channels.append(channel)

    # Create one large queue containing everything.
    queue = Queue.Queue()
    for channel in arclink_channels:
        queue.put(channel)
    # Use another queue for the successful downloads.
    successful_downloads = Queue.Queue()
    failed_downloads = Queue.Queue()
    my_threads = []
    # Launch 20 threads at max. Might seem a large number but timeout is set to
    # 60 seconds and they start with 1 second between each other.
    thread_count = min(20, len(arclink_channels))
    for _i in xrange(thread_count):
        thread = ArcLinkDownloadThread(
            queue=queue, successful_downloads=successful_downloads,
            failed_downloads=failed_downloads)
        my_threads.append(thread)
        thread.start()
        time.sleep(0.5)

    for thread in my_threads:
        thread.join()

    # Convert to list
    temp = []
    while failed_downloads.qsize():
        temp.append(failed_downloads.get())
    failed_downloads = temp
    # Now reassemble the original traces from the failed ArcLink downloads.
    # These will be downloaded as RESP files from IRIS>
    iris_channels = []
    for channel in channels:
        this_channel = deepcopy(channel)
        this_channel["channel"] = this_channel["channel"][:2] + "*"
        del this_channel["channel_id"]
        if this_channel in failed_downloads:
            iris_channels.append(this_channel)

    queue = Queue.Queue()
    for channel in iris_channels:
        queue.put(channel)

    # Now download the iris channels.
    # Launch 20 threads at max. Might seem a large number but timeout is set to
    # 60 seconds and they start with 1 second between each other.
    thread_count = min(20, len(iris_channels))
    for _i in xrange(thread_count):
        thread = IRISDownloadThread(
            queue=queue, successful_downloads=successful_downloads)
        my_threads.append(thread)
        thread.start()
        time.sleep(0.5)

    for thread in my_threads:
        thread.join()

    # Return the number of successfully downloaded files.
    return successful_downloads.qsize()
