#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os
import warnings

from .component import Component
from lasif import LASIFNotFoundError, LASIFWarning


class QueryComponent(Component):
    """
    This component is responsible for making queries across the different
    components and integrating them in a meaningful way.

    It should thus be initialized fairly late as it needs access to a number
    of other components via the communicator.
    """
    def get_stations_for_event(self, event_name):
        event = self.comm.events.get(event_name)
        waveforms = self.comm.waveforms.get_raw(event_name)
        station_coordinates = self.comm.stations.get_all_coordinates_at_time(
            event["origin_time"])

        stations = {}
        for waveform in waveforms:
            station_id = "%s.%s" % (waveform["network"], waveform["station"])
            if station_id in stations:
                continue

            try:
                stat_coords = station_coordinates[waveform["channel_id"]]
            except KeyError:
                # No station file for channel.
                continue

            # First attempt to retrieve from the station files.
            if stat_coords["latitude"] is not None:
                stations[station_id] = stat_coords
                continue
            # Then from the waveform file in the case of a sac file.
            elif waveform["latitude"] is not None:
                stations[station_id] = waveform
            # Otherwise



            if station_coordinates[waveform["channel_id"]]["latitude"] \
                    is not None:
                stations[station] = all_channel_coordinates[
                    waveform["channel_id"]]
                continue
            elif waveform["latitude"] is not None:
                stations[station] = waveform
            elif station in inv_db:
                stations[station] = inv_db[station]
            else:
                msg = ("Could not find coordinates for file '%s'. Will be "
                       "skipped" % waveform["filename"])
                warnings.warn(msg)
        return stations
