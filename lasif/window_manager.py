#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes managing misfit windows and associated operations.

This happens in a layer of hierarchies. It might look over-engineered but it
offers a nice separation of concerns and is pretty intuitive once it is
actually clear what is going on.

The :class:`~WindowGroupManager: class manages all windows per event and
iteration. This offers a convenient way to loop over all windows for an
event and a certain iteration. It will always return :class:`~WindowCollection`
objects.

A :class:`~WindowCollection` object manages all windows (for a given event
and iteration) for one channel. It contains :class:`~Window` object,
each defining a single window with an associated misfit and adjoint source.

Windows are serialized at the :class:`~WindowCollection` level so remember
to call :meth:`~WindowCollection.write` when adding/removing/changing windows.

One thing to keep in mind is that a window can be in several "states" for
lack of a better word. Each an every window will always be defined by a
start time, an end time, a weight normalized between 0.0 and 1.0, a tapering
function and the percentage of the data it tapers at each end, e.g. 0.5 for
a full width taper. A window with only these 5 components is an empty window:

.. code-block::xml

    <Window>
        <Starttime>2012-01-01T00:00:00.000000Z</Starttime>
        <Endtime>2012-01-01T00:01:00.000000Z</Endtime>
        <Weight>0.5</Weight>
        <Taper>cosine</Taper>
        <TaperPercentage>0.08</TaperPercentage>
    </Window>


A window can furthermore have an associated misfit functional, a simple
string denoting the type of misfit:

.. code-block::xml

    <Window>
        <Starttime>2012-01-01T00:00:00.000000Z</Starttime>
        <Endtime>2012-01-01T00:01:00.000000Z</Endtime>
        <Weight>0.5</Weight>
        <Taper>cosine</Taper>
        <TaperPercentage>0.08</TaperPercentage>
        <Misfit>some misfit</Misfit>
    </Window>


In the final state, the window can also contain details about the calculated
misfit. In that case it is always expected to have the value of the misfit.
All other parameters are optional and misfit specific.

.. code-block::xml

    <Window>
        <Starttime>2012-01-01T00:00:00.000000Z</Starttime>
        <Endtime>2012-01-01T00:01:00.000000Z</Endtime>
        <Weight>0.5</Weight>
        <Taper>cosine</Taper>
        <TaperPercentage>0.08</TaperPercentage>
        <Misfit>some misfit</Misfit>
        <MisfitDetails>
            <Value>1e-05</Value>
            <TimeDelay>-1.5</TimeDelay>
        </MisfitDetails>
    </Window>


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from glob import iglob
from obspy import UTCDateTime
from lxml import etree
from lxml.builder import E
import os


class WindowGroupManager(object):
    """
    Class managing all windows for one event and a certain iteration.
    """
    def __init__(self, directory, iteration, event_name):
        self._directory = directory
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)
        self._synthetic_tag = iteration
        self._event_name = event_name

    def __iter__(self):
        for channel_id in self.list():
            yield self.get(channel_id)

    def __len__(self):
        return len(self.list())

    def __str__(self):
         return (
             "WidowGroupManager containing windows for %i channels.\n"
             "\tEvent name: %s\n"
             "\tIteration: %s\n"
             "\tWindow directory: %s\n" % (
                 len(self), self._event_name, self._synthetic_tag,
                 os.path.relpath(self._directory)))

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
        """
        Returns a window collection object for the given channel.

        :param channel_id: The id of the channel in the form NET.STA.LOC.CHA
        """
        windowfile = self._get_window_filename(channel_id)
        if not os.path.exists(windowfile):
            return {}
        return WindowCollection(filename=windowfile)

    def get_windows_for_station(self, station_id):
        """
        Get a list of window collection objects for the given station.

        :param station_id: The id of the station in the form NET.STA
        """
        channels = [_i for _i in self.list()
                    if _i.startswith(station_id + ".")]
        windows = []
        for channel_id in channels:
            windows.append(self.get(channel_id))
        return windows

    def delete_windows_for_channel(self, channel_id):
        """
        Deletes all windows for a certain channel by removing the window
        file for the channel.

        :param channel_id: The channel id for the windows to delete in the
            form NET.STA.LOC.CHA
        """
        windowfile = self._get_window_filename(channel_id)
        if os.path.exists(windowfile):
            os.remove(windowfile)

    def _get_window_filename(self, channel_id):
        return os.path.join(self._directory, "window_%s.xml" % channel_id)


class WindowCollection(object):
    """
    Represents all the windows for one particular channel for one event and
    one iteration.
    """
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

    def __len__(self):
        return len(self.windows)

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
            if misfit_value is not None else None,
            collection=self))

    def delete_window(self, starttime, endtime, tolerance=0.01):
        """
        Deletes one or more windows from the group.

        A window is specified with its start- and endtime. A certain
        tolerance can also be given. The tolerance is the maximum allowed
        error if start- and endtime relative to the total time span.

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

        new_windows = []

        for window in self.windows:
            if (min_starttime <= window.starttime <= max_starttime) and \
                    (min_endtime <= window.endtime <= max_endtime):
                continue
            new_windows.append(window)
        self.windows = new_windows

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

        Will delete a possibly exiting file if the collection has no windows.

        :param filename: Path to save to. Will be overwritten if it already
            exists.
        """
        # A window collection that has no windows will attempt to remove its
        # own file if it has no windows without emitting a warnings.
        if not self.windows:
            try:
                os.remove(self.filename)
            except OSError:
                pass
            return

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


class Window(object):
    """
    Object representing one window.
    """
    __slots__ = ["starttime", "endtime", "weight", "taper",
                 "taper_percentage", "misfit", "__misfit_value",
                 "__collection"]

    def __init__(self, starttime, endtime, weight, taper,
                 taper_percentage, misfit, misfit_value, collection):
        self.starttime = UTCDateTime(starttime)
        self.endtime = UTCDateTime(endtime)
        self.weight = float(weight)
        self.taper = str(taper)
        taper_percentage = float(taper_percentage)
        if not 0.0 <= taper_percentage <= 0.5:
            raise ValueError("Invalid taper percentage.")
        self.taper_percentage = taper_percentage
        self.misfit = str(misfit)
        self.__misfit_value = float(misfit_value) \
            if misfit_value is not None else None
        # Reference to the window collection.
        self.__collection = collection

    @property
    def misfit_value(self):
        """
        Returns the misfit value. If not stored in the file, it will be
        calculated and thus is potentially an expensive operation.
        """
        return self.__misfit_value

    def get_adjoint_source(self):
        """
        Returns the adjoint source for the window. If not available it will
        be calculated and cached for future access. Calling this function
        will also set the misfit value of the window. Don't forget to write
        the window collection!
        """
        pass

    @property
    def ad_src_filename(self):
        """
        Filename of the adjoint source.
        """

    def __eq__(self, other):
        if not isinstance(other, Window):
            return False
        return self.starttime == other.starttime and \
               self.endtime == other.endtime and \
               self.weight == other.weight and \
               self.taper == other.taper and \
               self.taper_percentage == other.taper_percentage and \
               self.misfit == other.misfit and \
               self.__misfit_value == other.__misfit_value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return (
            "{duration:.2f} seconds window from {st}\n\tWeight: "
            "{weight:.2f}, {perc:.2f}% {taper} taper, {misfit} ({value})"

        ).format(
            duration=self.endtime - self.starttime,
            st=self.starttime,
            weight=self.weight,
            taper=self.taper,
            perc=self.taper_percentage * 100.0,
            misfit=self.misfit,
            value="%.3g" % self.__misfit_value
            if self.__misfit_value is not None else "not calculated")

    @property
    def length(self):
        """
        The length of the window in seconds.
        """
        return self.endtime - self.starttime

