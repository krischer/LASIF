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
        self._directory = directory
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)
        self._synthetic_tag = synthetic_tag
        self._event_name = event_name

    def __iter__(self):
        for channel_id in self.list():
            yield self.get(channel_id)

    def list(self):
        """
        Returns a list of station ids with windows.
        """
        windows = []
        for filename in iglob(os.path.join(self._directory, "window_*.xml")):
            windows.append(os.path.basename(filename).lstrip("window_")
                           .rstrip(".xml"))
        return sorted(windows)

    def get(self, channel_id):
        windowfile = self._get_window_filename(channel_id)
        if not os.path.exists(windowfile):
            return {}
        return WindowCollection(filename=windowfile)

    def get_windows_for_station(self, station_id):
        channels = [_i for _i in self.list()
                    if _i.startswith(station_id + ".")]
        windows = []
        for channel_id in channels:
            windows.append(self.get(channel_id))
        return windows

    def delete_windows(self, channel_id):
        """
        Deletes all windows for a certain channel.

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

        windows = self.get(channel_id)
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
        return os.path.join(self._directory, "window_%s.xml" % channel_id)


class Window(object):
    __slots__ = ["starttime", "endtime", "weight", "taper",
                 "taper_percentage", "misfit", "misfit_value"]

    def __init__(self, starttime, endtime, weight, taper,
                 taper_percentage, misfit, misfit_value):
        self.starttime = UTCDateTime(starttime)
        self.endtime = UTCDateTime(endtime)
        self.weight = float(weight)
        self.taper = str(taper)
        taper_percentage = float(taper_percentage)
        if not 0.0 <= taper_percentage <= 0.5:
            raise ValueError("Invalid taper percentage.")
        self.taper_percentage = taper_percentage
        self.misfit = str(misfit)
        self.misfit_value = float(misfit_value)

    def __eq__(self, other):
        if not isinstance(other, Window):
            return False
        return self.starttime == other.starttime and \
            self.endtime == other.endtime and \
            self.weight == other.weight and \
            self.taper == other.taper and \
            self.taper_percentage == other.taper_percentage and \
            self.misfit == other.misfit and \
            self.misfit_value == other.misfit_value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            "{duration:.2f} seconds window from {st}\n\tWeight: "
            "{weight:.2f}, {perc:.2f}% {taper} taper, {misfit}{value}"

        ).format(
            duration=self.endtime - self.starttime,
            st=self.starttime,
            weight=self.weight,
            taper=self.taper,
            perc=self.taper_percentage * 100.0,
            misfit=self.misfit,
            value=" (%.3g)" % self.misfit_value
                if self.misfit_value is not None else "")

    @property
    def duration(self):
        return self.endtime - self.starttime


class WindowCollection(object):
    def __init__(self, filename, windows=None, event_name=None,
                 channel_id=None, synthetics_tag=None):
        if windows and os.path.exists(filename):
            raise ValueError("An existing file and new windows is not "
                             "allowed. Either only a file or windows and a "
                             "non-existing file")
        if not os.path.exists(filename) and \
                None in [event_name, channel_id, synthetics_tag]:
            raise ValueError("If the file does not yet exist, "
                             "'event_name', 'channel_id', "
                             "and 'synthetics_tag' must all exist.")


        self.filename = filename
        self.event_name = event_name
        self.channel_id = channel_id
        self.synthetics_tag = synthetics_tag
        self.windows = []

        if os.path.exists(filename):
            self._parse()
        else:
            if windows:
                self.windows.extend(windows)

    def __eq__(self, other):
        if not isinstance(other, WindowCollection):
            return False
        return self.filename == other.filename and \
            self.event_name == other.event_name and \
            self.channel_id == other.channel_id and \
            self.synthetics_tag == other.synthetics_tag and \
            self.windows == other.windows

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        ret_str = ("Window group for channel '{channel_id}' with {count} "
                   "window(s):\n"
                   "  Event: {event_name}\n"
                   "  Iteration: {it}\n"
                   "  Window(s):\n    {windows}")
        return ret_str.format(count=len(self.windows), it=self.synthetics_tag,
                              channel_id=self.channel_id,
                              event_name=self.event_name,
                              windows="\n    ".join(
                                  ["* " + str(_i) for _i in self.windows]))

    def add_window(self, starttime, endtime, weight, taper,
                   taper_percentage, misfit, misfit_value):
        """
        Adds a single window.

        :param starttime: Starttime of the window.
        :param endtime: Endtime of the window.
        :param weight: Weight of the window.
        :param taper: Used taper.
        :param taper_percentage: Decimal percentage of taper at one end.
            Ranges from 0.0 to 0.5 for a full-width taper.
        :param misfit: Used misfit.
        :param misfit_value: Used misfit value.
        """
        self.windows.append(Window(
            starttime=UTCDateTime(starttime),
            endtime=UTCDateTime(endtime),
            weight=float(weight),
            taper=taper,
            taper_percentage=float(taper_percentage),
            misfit=misfit,
            misfit_value=float(misfit_value)
            if misfit_value is not None else None))

    def _parse(self):
        root = etree.parse(self.filename).getroot()
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
                        w["starttime"] = elem.text
                    elif elem.tag == "Endtime":
                        w["endtime"] = elem.text
                    elif elem.tag == "Weight":
                        w["weight"] = elem.text
                    elif elem.tag == "Taper":
                        w["taper"] = elem.text
                    elif elem.tag == "TaperPercentage":
                        w["taper_percentage"] = elem.text
                    elif elem.tag == "Misfit":
                        w["misfit"] = elem.text
                    elif elem.tag == "MisfitValue":
                        w["misfit_value"] = elem.text
                windows["windows"].append(w)

        self.event_name = windows["event"]
        self.channel_id = windows["channel_id"]
        self.synthetics_tag = windows["synthetic_tag"]

        for w in windows["windows"]:
            self.add_window(
                w["starttime"], w["endtime"], w["weight"], w["taper"],
                # Default to 0.05 to ease transition to the new format. The
                # old format did not track it at all. This will also trigger
                # a recalculation of the adjoint source in case no taper
                # percentage has been set before.
                w["taper_percentage"]  if "taper_percentage" in w else "0.05",
                w["misfit"],
                w["misfit_value"] if "misfit_value" in w else None)

    def write(self):
        """
        Writes the window group to the specified filename.

        :param filename: Path to save to. Will be overwritten if it already
            exists.
        """
        windows = []
        for w in self.windows:
            local_win = []
            local_win.append(E.Starttime(str(w.starttime)))
            local_win.append(E.Endtime(str(w.endtime)))
            local_win.append(E.Weight(str(w.weight)))
            local_win.append(E.Taper(str(w.taper)))
            local_win.append(E.TaperPercentage(str(w.taper_percentage)))
            local_win.append(E.Misfit(str(w.misfit)))
            if w.misfit_value is not None:
                local_win.append(E.MisfitValue(str(w.misfit_value)))
            windows.append(E.Window(*local_win))

        doc = (
            E.MisfitWindow(
                E.Event(self.event_name),
                E.ChannelID(self.channel_id),
                E.SyntheticsTag(self.synthetics_tag),
                *windows))
        with open(self.filename, "wb") as fh:
            fh.write(etree.tostring(doc, pretty_print=True,
                                    xml_declaration=True, encoding="utf-8"))
