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
import Queue
import StringIO
import threading
import time


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
    successful_downloads = 0

    class ArcLinkDownloadThread(threading.Thread):
        def __init__(self, queue):
            self.queue = queue
            super(ArcLinkDownloadThread, self).__init__()

        def run(self):
            global successful_downloads
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
                    memfile = StringIO.StringIO()
                    arc_client.saveResponse(memfile, channel["network"],
                        channel["station"], channel["location"],
                        channel["channel"], starttime=channel["starttime"],
                        endtime=channel["endtime"], format="mseed")
                except Exception as e:
                    memfile.close()
                    logger.error(str(e))
                    continue
                # XXX: Include sanity check.
                memfile.seek(0, 0)
                save_station_fct(memfile, channel["network"],
                    channel["station"], channel["location"],
                    channel["channel"], format="datalessSEED")
                if logger:
                    logger.info("Successfully downloaded dataless SEED for "
                        "channel %s.%s.%s.%s from ArcLink.") % (
                            channel["network"], channel["station"],
                            channel["location"], channel["channel"])
                successful_downloads += 1

    # Create one large queue containing everything.
    queue = Queue.Queue()
    for channel in channels:
        queue.put(channel)
    my_threads = []
    # Launch 20 threads at max. Might seem a large number but timeout is set to
    # 60 seconds and they start with 1 second between each other.
    thread_count = min(20, len(channels))
    for _i in xrange(thread_count):
        thread = ArcLinkDownloadThread(queue=queue)
        my_threads.append(thread)
        thread.start()
        time.sleep(1.0)

    for thread in my_threads:
        thread.join()

    return successful_downloads
