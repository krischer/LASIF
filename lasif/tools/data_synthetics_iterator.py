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


class DataSyntheticIterator(object):
    def __init__(self, comm, iteration_name, event_name):
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
        return self.comm.query.get_matching_waveforms(
            self.event_name, self.iteration, self.station_id, component=None)

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
