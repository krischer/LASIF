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
from lasif import LASIFNotFoundError


class DataSyntheticIterator(object):
    def __init__(self, comm, iteration, event):
        self.comm = comm
        self.event = self.comm.events.get(event)
        self.iteration = self.comm.iterations.get(iteration)

        self.event_name = self.event["event_name"]

        if self.event_name not in self.iteration.events:
            msg = "Event '%s' not part of iteration '%s'." % (
                self.event_name, self.iteration.name)
            raise LASIFNotFoundError(msg)

        # Get all stations defined for the given iteration and event.
        stations = set(self.iteration.events[self.event_name][
                       "stations"].keys())

        # Only use those stations that actually have processed and synthetic
        # data available! Especially synthetics might not always be available.
        processed = comm.waveforms.get_metadata_processed(
            self.event_name, self.iteration.processing_tag)
        synthetics = comm.waveforms.get_metadata_synthetic(
            self.event_name, self.iteration)
        processed = set(["%s.%s" % (_i["network"], _i["station"]) for _i in
                         processed])
        synthetics = set(["%s.%s" % (_i["network"], _i["station"]) for _i in
                         synthetics])
        self.stations = tuple(sorted(stations.intersection(
            processed).intersection(synthetics)))

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
            self.event_name, self.iteration, station_id)

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
