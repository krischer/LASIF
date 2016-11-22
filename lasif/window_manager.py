#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes managing misfit windows and associated operations.

This happens in a layer of hierarchies. It might look over-engineered but it
offers a nice separation of concerns and is pretty intuitive once it is
actually clear what is going on.

Furthermore it turned out to be way more complicated to manage windows and
the associated adjoint sources in a managable and flexible way than I
originally thought...

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
        <MisfitValue>0.05</MisfitValue>
        <MisfitDetails>
            <TimeDelay>-1.5</TimeDelay>
            <SomethingElse>complex</SomethingElse>
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

from lasif import LASIFAdjointSourceCalculationError

# XXX: Change this!
DEFAULT_AD_SRC_TYPE = "TimeFrequencyPhaseMisfitFichtner2008"


class WindowGroupManager(object):
    """
    Class managing all windows for one event and a certain iteration.
    """
    def __init__(self, directory, iteration, event_name, comm=None):
        """

        :param directory: The directory of the windows.
        :param iteration: The iteration name.
        :param event_name: The event name.
        :param comm: The communicator instance. Can be none, but required
            for some operations.
        """
        self._directory = directory
        if not os.path.exists(self._directory):
            os.makedirs(self._directory)
        self._synthetic_tag = iteration
        self._event_name = event_name
        self.comm = comm

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
            return WindowCollection(
                filename=windowfile, comm=self.comm,
                event_name=self._event_name, channel_id=channel_id,
                synthetics_tag=self._synthetic_tag)
        return WindowCollection(filename=windowfile, comm=self.comm)

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

    def delete_windows_for_station(self, station_id):
        """
        Deletes all windows for a certain station by removing all window
        files for the station.

        :param station_id: The station id for the windows to delete in the
            form NET.STA
        """
        channel_ids = [_i for _i in self.list()
                       if _i.startswith(station_id + ".")]
        for channel in channel_ids:
            self.delete_windows_for_channel(channel)

    def _get_window_filename(self, channel_id):
        return os.path.join(self._directory, "window_%s.xml" % channel_id)


class WindowCollection(object):
    """
    Represents all the windows for one particular channel for one event and
    one iteration.
    """
    def __init__(self, filename, windows=None, event_name=None,
                 channel_id=None, synthetics_tag=None, comm=None):
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
        self.comm = comm
        self.windows = []

        if os.path.exists(filename):
            self._parse()
        else:
            if windows:
                self.windows.extend(windows)

        if self.comm:
            self.event = self.comm.events.get(self.event_name)
            self.iteration = self.comm.iterations.get(self.synthetics_tag)

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

    def __iter__(self):
        return iter(self.windows)

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

    def add_window(self, starttime, endtime, weight=1.0, taper="cosine",
                   taper_percentage=0.05, misfit_type=None):
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
            starttime=starttime,
            endtime=endtime,
            weight=weight,
            taper=taper,
            taper_percentage=taper_percentage,
            misfit_type=misfit_type,
            collection=self))

    def plot(self, show=True, filename=None):
        """
        Plots the windows for the object's channel.

        :param show: Calls ``plt.show()`` before returning if True.
        :param filename: If given, a picture will be saved to this path.

        :return: The possibly created axes object.
        """
        if self.comm is None:
            raise ValueError("Operation only possible with an active "
                             "communicator instance.")

        import matplotlib.pylab as plt

        ax = self.comm.visualizations.plot_data_and_synthetics(
            self.event, self.iteration, self.channel_id, show=False)
        ylim = ax.get_ylim()

        st = self.event["origin_time"]
        for window in self.windows:
            ax.fill_between((window.starttime - st, window.endtime - st),
                            [ylim[0] - 10] * 2, [ylim[1] + 10] * 2,
                            alpha=0.3, lw=0, zorder=-10)
        ax.set_ylim(*ylim)
        if show:
            plt.tight_layout()
            plt.show()
            plt.close()
        elif filename:
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        return ax

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
                w["taper_percentage"] if "taper_percentage" in w else "0.05",
                w["misfit"] if "misfit" in w else None)

    def write(self):
        """
        Writes the window group to the specified filename.

        Will delete a possibly exiting file if the collection has no windows.
        """
        # A window collection that has no windows will attempt to remove its
        # own file if it has no windows without emitting a warnings.
        if not self.windows:
            try:
                os.remove(self.filename)
            except OSError:
                pass
            return

        d = os.path.dirname(self.filename)
        if not os.path.exists(d):
            os.makedirs(d)

        windows = []
        for w in self.windows:
            local_win = []
            local_win.append(E.Starttime(str(w.starttime)))
            local_win.append(E.Endtime(str(w.endtime)))
            local_win.append(E.Weight(str(w.weight)))
            local_win.append(E.Taper(str(w.taper)))
            local_win.append(E.TaperPercentage(str(w.taper_percentage)))
            if w.misfit_type is not None:
                local_win.append(E.Misfit(str(w.misfit_type)))
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
    __slots__ = ["starttime", "endtime", "weight", "taper",
                 "taper_percentage", "misfit_type", "__collection", "comm"]

    def __init__(self, starttime, endtime, weight, taper,
                 taper_percentage, misfit_type=None,
                 collection=None):
        """
        Object representing one window.

        :param starttime: The start time of the window.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: The end time of the window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param weight: The weight of the window normalized between 0.0 and 1.0
        :type weight: float
        :param taper: The type of taper, must coincide with an ObsPy taper.
        :type taper: str
        :param taper_percentage: The one sided taper percentage between 0.0
            and 0.5. A value of 0.0 taper nothing, 0.5 is a full width
            taper, e.g. 50 % at each side.
        :type taper_percentage: float
        :param misfit_type: The misfit (and adjoint source) type as a string.
        :type misfit_type: str, optional
        :param collection: The window collection object. Necessary for
            example to compute misfit values and adjoint sources on demand
            for a given window. The window itself does not have enough
            information to do that.
        :type collection: :class:`~.WindowCollection`, optional
        """
        # Force types of all values.
        self.starttime = UTCDateTime(starttime)
        self.endtime = UTCDateTime(endtime)
        self.weight = float(weight)
        self.taper = str(taper)
        self.taper_percentage = float(taper_percentage)
        # Some sanity checks.
        if self.starttime >= self.endtime:
            raise ValueError("The window starttime must be smaller than the "
                             "window endtime.")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Window weight must be between 0.0 and 1.0.")
        if not 0.0 <= self.taper_percentage <= 0.5:
            raise ValueError("Invalid taper percentage. Must be between 0.0 "
                             "and 0.5")

        # All other values are optional.
        self.misfit_type = str(misfit_type) \
            if misfit_type is not None else None

        # Reference to the window collection.
        self.__collection = collection
        if collection is not None:
            self.comm = collection.comm
        else:
            self.comm = None

    @property
    def adjoint_source(self):
        adj_src = self.get_adjoint_source()
        if adj_src["adjoint_source"] is None:
            raise LASIFAdjointSourceCalculationError("Could not calculate "
                                                     "adjoint source!")
        return adj_src

    def get_adjoint_source(self):
        if self.comm is None:
            raise ValueError("Operation only possible with an active "
                             "communicator instance.")
        if self.misfit_type is None:
            self.misfit_type = DEFAULT_AD_SRC_TYPE
        adsrc = self.comm.adjoint_sources.calculate_adjoint_source(
            self.__collection.event_name, self.__collection.synthetics_tag,
            self.__collection.channel_id,
            self.starttime, self.endtime, self.taper,
            self.taper_percentage, self.misfit_type)
        return adsrc

    def plot_adjoint_source(self):
        if self.comm is None:
            raise ValueError("Operation only possible with an active "
                             "communicator instance.")
        if self.misfit_type is None:
            self.misfit_type = DEFAULT_AD_SRC_TYPE

        import matplotlib.pyplot as plt
        plt.close("all")
        plt.figure(figsize=(15, 10))

        self.comm.adjoint_sources.calculate_adjoint_source(
            self.__collection.event_name, self.__collection.synthetics_tag,
            self.__collection.channel_id,
            self.starttime, self.endtime, self.taper,
            self.taper_percentage, self.misfit_type, plot=True)

        plt.show()

    @property
    def misfit_details(self):
        """
        Returns the misfit details. If not stored in the file, it will be
        calculated and thus is potentially an expensive operation.
        """
        adj_src = self.get_adjoint_source()
        return adj_src["details"]

    @property
    def misfit_value(self):
        adj_src = self.get_adjoint_source()
        return adj_src["misfit_value"]

    def __eq__(self, other):
        if not isinstance(other, Window):
            return False
        # No need to check the actual details as they by definition must be
        # identical if everything else is identical.
        return self.starttime == other.starttime and \
            self.endtime == other.endtime and \
            self.weight == other.weight and \
            self.taper == other.taper and \
            self.taper_percentage == other.taper_percentage and \
            self.misfit_type == other.misfit_type

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        if self.misfit_type is None:
            misfit_str = ""
        elif self.misfit_value is None:
            misfit_str = ", %s (not calculated)" % self.misfit_type
        else:
            misfit_str = ", %s (%.3g)" % (self.misfit_type, self.misfit_value)
        return (
            "{duration:.2f} seconds window from {st}\n\tWeight: "
            "{weight:.2f}, {perc:.2f}% {taper} taper{misfit_str}"

        ).format(
            duration=self.endtime - self.starttime,
            st=self.starttime,
            weight=self.weight,
            taper=self.taper,
            perc=self.taper_percentage * 100.0,
            misfit_str=misfit_str)

    @property
    def length(self):
        """
        The length of the window in seconds.
        """
        return self.endtime - self.starttime
