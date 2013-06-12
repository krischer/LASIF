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
from obspy import UTCDateTime
from lxml import etree
from lxml.builder import E
import os


class MisfitWindowManager(object):
    def __init__(self, directory, synthetic_tag, event_name):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.synthetic_tag = synthetic_tag
        self.event_name = event_name

    def get_windows(self, channel_id):
        windowfile = self._get_window_filename(channel_id)
        if not os.path.exists(windowfile):
            return {}
        return self._read_windowfile(windowfile)

    def delete_windows(self, channel_id):
        windowfile = self._get_window_filename(channel_id)
        if os.path.exists(windowfile):
            os.remove(windowfile)

    def _get_window_filename(self, channel_id):
        return os.path.join(self.directory, "window_%s.xml" % channel_id)

    def _read_windowfile(self, filename):
        root = etree.parse(filename).getroot()
        windows = {"windows": []}
        for element in root:
            if element.tag == "Event":
                windows["event"] = element.text
                continue
            elif element.tag == "ChannelId":
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
                E.ChannelId(window["channel_id"]),
                E.SyntheticsTag(window["synthetic_tag"]),
                *windows))
        with open(windowfile, "wb") as fh:
            fh.write(etree.tostring(doc, pretty_print=True,
                xml_declaration=True, encoding="utf-8"))
