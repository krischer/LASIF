#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two way data synthetics iterator.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import numpy as np
from obspy import read, Stream
import warnings


class TwoWayIter(object):
    """
    A two way iterator returning a dictionary with processed data and
    synthetics.
    """
    def __init__(self, project, stations, processed_waveforms, event_name,
                 iteration_name):
        """
        :param project: A LASIF Project instance.
        :param stations: A list of stations available for synthetic and
            processed data.
        :param processed_waveforms: A list of dictionaries containing
            information about processed waveform files.
        :param event_name: The event name as understandable by a LASIF project.
        :param iteration_name: The iteration name name as understandable by a
            LASIF project.
        """
        self.items = stations.items()
        self.current_index = -1
        self.project = project
        self.waveforms = processed_waveforms
        self.event_name = event_name
        self.iteration_name = iteration_name

    def __iter__(self):
        return self

    def next(self):
        """
        Called to retrieve the next item.

        Raises a StopIteration exception when the end has been reached.
        """
        self.current_index += 1
        if self.current_index > (len(self.items) - 1):
            self.current_index = len(self.items) - 1
            raise StopIteration
        return self.get_value()

    def prev(self):
        """
        Called to retrieve the previous item.

        Raises a StopIteration exception when the beginning has been
        reached.
        """
        self.current_index -= 1
        if self.current_index < 0:
            self.current_index = 0
            raise StopIteration
        return self.get_value()

    def get_value(self):
        """
        Return the value for the currently set index.
        """
        station_id, coordinates = self.items[self.current_index]

        # Get the processed data.
        data = Stream()
        # Now get the actual waveform files. Also find the
        # corresponding station file and check the coordinates.
        this_waveforms = {
            _i["channel_id"]: _i for _i in self.waveforms
            if _i["channel_id"].startswith(station_id + ".")}
        for key, value in this_waveforms.iteritems():
            data += read(value["filename"])[0]
        if not this_waveforms:
            msg = "Could not retrieve data for station '%s'." % \
                station_id
            # Always raise the warning!
            warnings.warn_explicit(
                msg, UserWarning, __file__,
                inspect.currentframe().f_back.f_lineno)
            return None

        # Get the synthetics.
        try:
            synthetics = self.project._get_and_rotate_synthetics(
                self.event_name, station_id, self.iteration_name)
        except ValueError:
            msg = "No synthetics found for station '%s'" % station_id
            warnings.warn_explicit(
                msg, UserWarning, __file__,
                inspect.currentframe().f_back.f_lineno)
            return None

        # Scale the data to the synthetics.
        for data_tr in data:
            # Explicit conversion to floating points.
            data_tr.data = np.require(data_tr.data, dtype=np.float32)
            synthetic_tr = synthetics.select(
                component=data_tr.stats.channel[-1])[0]
            scaling_factor = synthetic_tr.data.ptp() / \
                data_tr.data.ptp()
            # Store and apply the scaling.
            data_tr.stats.scaling_factor = scaling_factor
            data_tr.data *= scaling_factor

        return {"data": data, "synthetics": synthetics,
                "coordinates": coordinates}
