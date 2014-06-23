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

from lasif import LASIFNotFoundError

DataTuple = collections.namedtuple("DataTuple", ["data", "synthetics",
                                                 "coordinates"])

class DataSyntheticIterator(object):
    def __init__(self, comm, iteration_name, event_name, scale_data=False):
        self.comm = comm

        self.event_name = event_name
        self.iteration_name = iteration_name
        self.long_iteration_name = \
            self.comm.iterations.get_long_iteration_name(iteration_name)

        self.iteration = self.comm.iterations.get(iteration_name)

        if event_name not in self.iteration.events:
            msg = "Event '%s' not part of iteration '%s'." % (
                event_name, iteration_name)
            raise LASIFNotFoundError(msg)

        self._scale_data_flag = scale_data

        # Get all stations defined for the given iteration and event.
        self.stations = tuple(sorted(
            self.iteration.events[event_name]["stations"].keys()))

        self._current_index = -1

    def __len__(self):
        return len(self.stations)

    def __iter__(self):
        return self

    def reset(self):
        """
        Resets the iterator, e.g. sets the index to the start.
        """
        self._current_index = -1

    def get(self, station_id):
        # Get the metadata for the processed and synthetics for this
        # particular station.
        data = self.comm.waveforms.get_waveforms_processed(
            self.event_name, station_id,
            tag=self.iteration.get_processing_tag())
        synthetics = self.comm.waveforms.get_waveforms_synthetic(
            self.event_name, station_id,
            long_iteration_name=self.long_iteration_name)
        coordinates = self.comm.query.get_coordinates_for_station(
            self.event_name, station_id)

        if self._scale_data_flag:
            for data_tr in data:
                synthetic_tr = synthetics.select(
                    channel=data_tr.stats.channel[-1])[0]
                scaling_factor = synthetic_tr.data.ptp() / \
                                 data_tr.data.ptp()
                # Store and apply the scaling.
                data_tr.stats.scaling_factor = scaling_factor
                data_tr.data *= scaling_factor

        return DataTuple(data=data, synthetics=synthetics,
                         coordinates=coordinates)

    def next(self):
        """
        Called to retrieve the next item.

        Raises a StopIteration exception when the end has been reached.
        """
        self._current_index += 1
        if self._current_index > (len(self) - 1):
            self._current_index = len(self) - 1
            raise StopIteration
        return self.get(self.stations[self._current_index])

    def prev(self):
        """
        Called to retrieve the previous item.

        Raises a StopIteration exception when the beginning has been
        reached.
        """
        self._current_index -= 1
        if self._current_index < 0:
            self._current_index = 0
            raise StopIteration
        return self.get(self.stations[self._current_index])
