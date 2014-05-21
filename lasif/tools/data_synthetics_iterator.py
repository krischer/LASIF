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
import collections
import inspect
import numpy as np
import warnings

from lasif import LASIFException

DataTuple = collections.namedtuple("DataTuple", ["data", "synthetics",
                                                 "coordinates"])


class DataSyntheticIterator(object):
    """
    An object enabling the iteration over all processed waveforms and the
    corresponding synthetics for one iteration and event.

    It provides next() and previous() method
    """
    def __init__(self, project, event_name, iteration_name):
        """
        :param project: The LASIF project instance.
        :param event_name: The name of the event.
        :param iteration_name: The short name of the iteration.
        """
        self.current_index = -1
        self._project = project
        self.event_name = event_name

        self.iteration = self._project._get_iteration(iteration_name)
        self._processing_tag = self.iteration.get_processing_tag()

        # Make sure the event is defined for the given iteration.
        if event_name not in self.iteration.events:
            msg = "Event '%s' not used in iteration '%s.'" % \
                  (event_name, iteration_name)
            raise LASIFException(msg)

        # Get the coordinates.
        self._station_coordinates = \
            self._project.get_stations_for_event(event_name)
        # Get all stations with a defined weight.
        self.station_keys = [
            key for key, value in
            self.iteration.events[event_name]["stations"].iteritems()
            if value["station_weight"] and key in self._station_coordinates]

    def __len__(self):
        return len(self.station_keys)

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator, e.g. sets the index to the start.
        """
        self.current_index += -1

    def next(self):
        """
        Called to retrieve the next item.

        Raises a StopIteration exception when the end has been reached.
        """
        self.current_index += 1
        if self.current_index > (len(self.station_keys) - 1):
            self.current_index = len(self.station_keys) - 1
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
        station_id = self.station_keys[self.current_index]

        # Get the data.
        try:
            data = self._project.get_waveform_data(
                self.event_name, station_id, data_type="processed",
                tag=self._processing_tag)
        except LASIFException:
            msg = "No data found for station '%s'." % station_id
            warnings.warn_explicit(
                msg, UserWarning, __file__,
                inspect.currentframe().f_back.f_lineno)
            return None

        # Get the synthetics.
        try:
            synthetics = self._project.get_waveform_data(
                self.event_name, station_id, data_type="synthetic",
                iteration_name=self.iteration.iteration_name)
        except LASIFException:
            msg = "No synthetics found for station '%s'." % station_id
            warnings.warn_explicit(
                msg, UserWarning, __file__,
                inspect.currentframe().f_back.f_lineno)
            return None

        # Scale the data to the synthetics.
        for data_tr in data:
            # Explicit conversion to floating points.
            data_tr.data = np.require(data_tr.data, dtype=np.float32)
            synthetic_tr = synthetics.select(
                channel=data_tr.stats.channel[-1])[0]
            scaling_factor = synthetic_tr.data.ptp() / \
                data_tr.data.ptp()
            # Store and apply the scaling.
            data_tr.stats.scaling_factor = scaling_factor
            data_tr.data *= scaling_factor

        return DataTuple(data, synthetics,
                         self._station_coordinates[station_id])
