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
        def __init__(self, queue, counter):
            self.queue = queue
            self.counter = counter
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
                for _i in xrange(10):
                    try:
                        arc_client = obspy.arclink.Client(user=arclink_user,
                            timeout=60)
                        success = True
                        break
                    except:
                        time.sleep(0.3)
                if success is False:
                    msg = (" A problem occured initializing ArcLink. Try "
                        "again later")
                    logger.error(msg)
                    continue
                try:
                    memfile = StringIO.StringIO()
                    arc_client.saveResponse(memfile, channel["network"],
                        channel["station"], channel["location"],
                        channel["channel"], starttime=channel["starttime"],
                        endtime=channel["endtime"], format="SEED")
                except Exception as e:
                    msg = "While downloading %s [%s to %s]: %s" % (
                        channel_id, channel["starttime"], channel["endtime"],
                        str(e))
                    logger.error(msg)
                    continue
                memfile.seek(0, 0)
                # Read the file again and perform a sanity check.
                try:
                    parser = Parser(memfile)
                except:
                    msg = ("Arclink did not return a valid dataless SEED file "
                        "for channel %s [%s-%s]") % (channel_id, starttime,
                        endtime)
                    logger.error(msg)
                    continue
                if not utils.channel_in_parser(parser, channel_id, starttime,
                        endtime):
                    msg = ("Arclink returned a valid dataless SEED file "
                        "for channel %s [%s to %s], but it does not actually "
                        " contain data for the requested channel and time "
                        "frame.") % \
                        (channel_id, starttime, endtime)
                    logger.error(msg)
                    continue
                memfile.seek(0, 0)
                save_station_fct(memfile, channel["network"],
                    channel["station"], channel["location"],
                    channel["channel"], format="datalessSEED")
                counter.put(True)
                if logger:
                    logger.info("Successfully downloaded dataless SEED for "
                        "channel %s.%s.%s.%s from ArcLink." % (
                            channel["network"], channel["station"],
                            channel["location"], channel["channel"]))

    # Create one large queue containing everything.
    queue = Queue.Queue()
    for channel in channels:
        queue.put(channel)
    # Also use a queue for the counter. Slightly awkward but apparently it is
    # savest to use a Queue or dequeue in multi threaded parts.
    counter = Queue.Queue()
    my_threads = []
    # Launch 20 threads at max. Might seem a large number but timeout is set to
    # 60 seconds and they start with 1 second between each other.
    thread_count = min(20, len(channels))
    for _i in xrange(thread_count):
        thread = ArcLinkDownloadThread(queue=queue, counter=counter)
        my_threads.append(thread)
        thread.start()
        time.sleep(1.0)

    for thread in my_threads:
        thread.join()

    # Return the number of successfully downloaded files.
    return counter.qsize()
