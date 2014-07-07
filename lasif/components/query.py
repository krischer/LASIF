#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import os

from lasif import LASIFNotFoundError

from .component import Component


class QueryComponent(Component):
    """
    This component is responsible for making queries across the different
    components and integrating them in a meaningful way.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.

    It should thus be initialized fairly late as it needs access to a number
    of other components via the communicator.
    """
    def get_all_stations_for_event(self, event_name):
        """
        Returns a list of all stations for one event.

        A station is considered to be available for an event if at least one
        channel has raw data and an associated station file. Furthermore it
        must be possible to derive coordinates for the station.

        :type event_name: str
        :param event_name: Name of the event.

        >>> import pprint
        >>> comm = getfixture('query_comm')
        >>> pprint.pprint(comm.query.get_all_stations_for_event(
        ...     "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")) \
        # doctest: +NORMALIZE_WHITESPACE
        {u'HL.ARG': {'elevation_in_m': 170.0, 'latitude': 36.216,
                     'local_depth_in_m': 0.0, 'longitude': 28.126},
         u'HT.SIGR': {'elevation_in_m': 93.0, 'latitude': 39.2114,
                      'local_depth_in_m': 0.0, 'longitude': 25.8553},
         u'KO.KULA': {'elevation_in_m': 915.0, 'latitude': 38.5145,
                      'local_depth_in_m': 0.0, 'longitude': 28.6607},
         u'KO.RSDY': {'elevation_in_m': 0.0, 'latitude': 40.3972,
                      'local_depth_in_m': 0.0, 'longitude': 37.3273}}


        Raises a :class:`~lasif.LASIFNotFoundError` if the event does not
        exist.

        >>> comm.query.get_all_stations_for_event("RandomEvent")
        Traceback (most recent call last):
            ...
        LASIFNotFoundError: ...
        """
        event = self.comm.events.get(event_name)

        # Collect information from all the different places.
        waveform_metadata = self.comm.waveforms.get_metadata_raw(event_name)
        station_coordinates = self.comm.stations.get_all_channels_at_time(
            event["origin_time"])
        inventory_coordinates = self.comm.inventory_db.get_all_coordinates()

        stations = {}
        for waveform in waveform_metadata:
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
            # Then from the waveform metadata in the case of a sac file.
            elif waveform["latitude"] is not None:
                stations[station_id] = waveform
                continue
            # If that still does not work, check if the inventory database
            # has an entry.
            elif station_id in inventory_coordinates:
                coords = inventory_coordinates[station_id]
                # Otherwise already queried for, but no coordinates found.
                if coords["latitude"]:
                    stations[station_id] = coords
                continue

            # The last resort is a new query via the inventory database.
            coords = self.comm.inventory_db.get_coordinates(station_id)
            if coords["latitude"]:
                stations[station_id] = coords
        return stations

    def get_coordinates_for_station(self, event_name, station_id):
        """
        Get the coordinates for one station.

        Must be in sync with :func:`~.get_all_stations_for_event`.
        """
        event = self.comm.events.get(event_name)

        # Collect information from all the different places.
        waveform = self.comm.waveforms.get_metadata_raw_for_station(
            event_name, station_id)[0]
        station_coordinates = self.comm.stations.get_all_channels_at_time(
            event["origin_time"])

        try:
            stat_coords = station_coordinates[waveform["channel_id"]]
        except KeyError:
            # No station file for channel.
            raise LASIFNotFoundError("Station '%s' has no available "
                                      "station file." % station_id)

        # First attempt to retrieve from the station files.
        if stat_coords["latitude"] is not None:
            return stat_coords
        # Then from the waveform metadata in the case of a sac file.
        elif waveform["latitude"] is not None:
            return waveform
        # The last resort is a new query via the inventory database.
        coords = self.comm.inventory_db.get_coordinates(station_id)
        if coords["latitude"]:
            return coords
        else:
            # No station file for channel.
            raise LASIFNotFoundError("Coordinates could not be deduced for "
                                     "station '%s' and event '%s'." % (
                                     station_id, event_name))

    def get_stations_for_all_events(self):
        """
        Returns a dictionary with a list of stations per event.
        """
        events = {}
        for event in self.comm.events.list():
            try:
                data = self.get_all_stations_for_event(event).keys()
            except LASIFNotFoundError:
                continue
            events[event] = data
        return events

    def get_iteration_status(self, iteration_name):
        """
        Return the status of an iteration. This is a query command as it
        integrates a lot of different information.
        """
        iteration = self.comm.iterations.get(iteration_name)

        events = collections.defaultdict(dict)

        # Get all the data.
        for event_name, event_dict in iteration.events.iteritems():
            # Get all stations that should be defined for the current
            # iteration.
            stations = set(event_dict["stations"].keys())

            # Raw data.
            try:
                raw = self.comm.waveforms.get_metadata_raw(event_name)
                # Get a list of all stations
                raw = set((
                    "{network}.{station}".format(**_i) for _i in raw))
                # Get all stations in raw that are also defined for the
                # current iteration.
                raw_stations = stations.intersection(raw)
            except LASIFNotFoundError:
                raw_stations = {}
            events[event_name]["raw"] = raw_stations

            # Processed data.
            try:
                processed = self.comm.waveforms.get_metadata_processed(
                    event_name, iteration.get_processing_tag())
                # Get a list of all stations
                processed = set((
                    "{network}.{station}".format(**_i) for _i in processed))
                # Get all stations in raw that are also defined for the
                # current iteration.
                processed_stations = stations.intersection(processed)
            except LASIFNotFoundError:
                processed_stations = {}
            events[event_name]["processed"] = processed_stations

            # Synthetic data.
            try:
                synthetic = self.comm.waveforms.get_metadata_synthetic(
                    event_name,
                    self.comm.iterations.get_long_iteration_name(
                        iteration_name))
                # Get a list of all stations
                synthetic = set((
                    "{network}.{station}".format(**_i) for _i in synthetic))
                # Get all stations in raw that are also defined for the
                # current iteration.
                synthetic_stations = stations.intersection(synthetic)
            except LASIFNotFoundError:
                synthetic_stations = {}
            events[event_name]["synthetic"] = synthetic_stations

        return dict(events)

    def get_data_and_synthetics_iterator(self, iteration_name, event_name,
                                         scale_data=False):
        """
        Get the processed data and matching synthetics for a particular event.
        """
        from ..tools.data_synthetics_iterator import DataSyntheticIterator
        return DataSyntheticIterator(self.comm, iteration_name, event_name,
                                     scale_data)

    def discover_available_data(self, event_name, station_id):
        """
        Discovers the available data for one event at a certain station.

        Will raise a :exc:`~lasif.LASIFNotFoundError` if no raw data is
        found for the given event and station combination.

        :type event_name: str
        :param event_name: The name of the event.
        :type station_id: str
        :param station_id: The id of the station in question.

        :rtype: dict
        :returns: Return a dictionary with "processed" and "synthetic" keys.
            Both values will be a list of strings. In the case of "processed"
            it will be a list of all available preprocessing tags. In the case
            of the synthetics it will be a list of all iterations for which
            synthetics are available.
        """
        if not self.comm.events.has_event(event_name):
            msg = "Event '%s' not found in project." % event_name
            raise LASIFNotFoundError(msg)
        # Attempt to get the station coordinates. This ensures availability
        # of the raw data.
        self.get_coordinates_for_station(event_name, station_id)

        def get_components(waveform_cache):
            return sorted([_i["channel"][-1] for _i in waveform_cache],
                          reverse=True)

        raw = self.comm.waveforms.get_metadata_raw_for_station(event_name,
                                                               station_id)
        raw_comps = get_components(raw)

        # Collect all tags and iteration names.
        all_files = {
            "raw": {"raw": raw_comps},
            "processed": {},
            "synthetic": {}}

        # Get the available synthetic and processing tags.
        proc_tags = self.comm.waveforms.get_available_processing_tags(
            event_name)
        iterations = self.comm.waveforms.get_available_synthetics(event_name)

        for tag in proc_tags:
            procs = self.comm.waveforms.get_metadata_processed_for_station(
                event_name, tag, station_id)
            comps = get_components(procs)
            if not comps:
                continue
            all_files["processed"][tag] = comps

        synthetic_coordinates_mapping = {"X": "N",
                                         "Y": "E",
                                         "Z": "Z",
                                         "N": "N",
                                         "E": "E"}
        for it in iterations:
            its = self.comm.waveforms.get_metadata_synthetic_for_station(
                event_name, it, station_id)
            comps = get_components(its)
            if not comps:
                continue
            comps = [synthetic_coordinates_mapping[i] for i in comps]
            all_files["synthetic"][it] = sorted(comps, reverse=True)
        return all_files
