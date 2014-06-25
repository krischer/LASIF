#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A class handling misfit windows based on a channel_id.

A single misfit window is identified by:
    * Starttime
    * Endtime
    * Taper
    * Misfit Type

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from copy import deepcopy
from glob import iglob
from obspy import UTCDateTime
from lxml import etree
from lxml.builder import E
import os


class MisfitWindowManager(object):
    """
    A simple class reading and writing Windows.
    """
    def __init__(self, directory, synthetic_tag, event_name):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.synthetic_tag = synthetic_tag
        self.event_name = event_name

    def __iter__(self):
        for filename in iglob(os.path.join(self.directory, "window_*.xml")):
            yield self._read_windowfile(filename)

    def get_windows(self, channel_id):
        windowfile = self._get_window_filename(channel_id)
        if not os.path.exists(windowfile):
            return {}
        return self._read_windowfile(windowfile)

    def get_windows_for_station(self, station_id):
        files = iglob(os.path.join(self.directory, "window_%s.*.*.xml" %
                      station_id))
        windows = []
        for filename in files:
            windows.append(self._read_windowfile(filename))
        return windows

    def list_windows(self):
        """
        Get a list of channels with windows.
        """
        files = iglob(os.path.join(self.directory, "window_*.xml"))
        return [os.path.basename(i).lstrip("window_").rstrip(".xml") for i in
                files]

    def delete_windows(self, channel_id):
        """
        Deletes all windows for a certain channal.

        In essence it will just delete the file.

        :param channel_id: The channel id for the windows to delete.
        """
        windowfile = self._get_window_filename(channel_id)
        if os.path.exists(windowfile):
            os.remove(windowfile)

    def delete_window(self, channel_id, starttime, endtime, tolerance=0.01):
        """
        Deletes one specific window.

        If it is the only window in the file, the file will be deleted. A
        window is specified with the channel_id, start- and endtime. A certain
        tolerance can also be given. The tolerance is the maximum allowed error
        if start- and endtime relative to the total time span.

        :param channel_id: The channel id of the window to be deleted.
        :param starttime: The starttime of the window to be deleted.
        :param endtime: The endtime of the window to be deleted.
        :param tolerance: The maximum acceptable deviation in start- and
            endtime relative to the total timespan. Defaults to 0.01, e.g. 1%.
        """
        tolerance = tolerance * (endtime - starttime)
        min_starttime = starttime - tolerance
        max_starttime = starttime + tolerance
        min_endtime = endtime - tolerance
        max_endtime = endtime + tolerance

        windows = self.get_windows(channel_id)
        if not windows:
            msg = "Window not found."
            raise ValueError(msg)

        new_windows = deepcopy(windows)
        new_windows["windows"] = []
        found_window = False

        for window in windows["windows"]:
            if (min_starttime <= window["starttime"] <= max_starttime) and \
                    (min_endtime <= window["endtime"] <= max_endtime):
                found_window = True
                continue
            new_windows["windows"].append(window)

        if found_window is False:
            msg = "Window not found."
            raise ValueError(msg)

        # Delete the file if no windows are left.
        if not new_windows["windows"]:
            self.delete_windows(channel_id)
            return

        self._write_window(channel_id, new_windows)

    def _get_window_filename(self, channel_id):
        return os.path.join(self.directory, "window_%s.xml" % channel_id)

    def _read_windowfile(self, filename):
        root = etree.parse(filename).getroot()
        windows = {"windows": []}
        for element in root:
            if element.tag == "Event":
                windows["event"] = element.text
                continue
            elif element.tag == "ChannelID":
                windows["channel_id"] = element.text
                continue
            elif element.tag == "SyntheticsTag":
                windows["synthetic_tag"] = element.text
                continue
            elif element.tag == "Window":
                w = {}
                for elem in element:
                    if elem.tag == "Starttime":
                        w["starttime"] = UTCDateTime(elem.text)
                    elif elem.tag == "Endtime":
                        w["endtime"] = UTCDateTime(elem.text)
                    elif elem.tag == "Weight":
                        w["weight"] = float(elem.text)
                    elif elem.tag == "Taper":
                        w["taper"] = elem.text
                    elif elem.tag == "Misfit":
                        w["misfit"] = elem.text
                windows["windows"].append(w)
        return windows

    def write_window(self, channel_id, starttime, endtime, weight, taper,
                     misfit):
        window = self.get_windows(channel_id)
        if not window:
            window["event"] = self.event_name
            window["channel_id"] = channel_id
            window["synthetic_tag"] = self.synthetic_tag
            window["windows"] = []
        window["windows"].append({
            "starttime": starttime,
            "endtime": endtime,
            "weight": weight,
            "taper": taper,
            "misfit": misfit})
        self._write_window(channel_id, window)

    def _write_window(self, channel_id, window):
        windowfile = self._get_window_filename(channel_id)
        windows = [
            E.Window(
                E.Starttime(str(_i["starttime"])),
                E.Endtime(str(_i["endtime"])),
                E.Weight(str(_i["weight"])),
                E.Taper(_i["taper"]),
                E.Misfit(_i["misfit"])) for _i in window["windows"]]

        doc = (
            E.MisfitWindow(
                E.Event(window["event"]),
                E.ChannelID(window["channel_id"]),
                E.SyntheticsTag(window["synthetic_tag"]),
                *windows))
        with open(windowfile, "wb") as fh:
            fh.write(etree.tostring(doc, pretty_print=True,
                                    xml_declaration=True, encoding="utf-8"))
