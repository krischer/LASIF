#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import warnings

from lasif import LASIFError, LASIFNotFoundError, LASIFWarning
import pyasdf

from .component import Component

DataTuple = collections.namedtuple("DataTuple", ["data", "synthetics",
                                                 "coordinates"])


class QueryComponent(Component):
    """
    This component is responsible for making queries across the different
    components and integrating them in a meaningful way.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.

    It should thus be initialized fairly late as it needs access to a number
    of other components via the communicator.
    """
    def get_all_stations_for_event(self, event_name, list_only=False):
        """
        Returns a dictionary of all stations for one event and their
        coordinates.

        A station is considered to be available for an event if at least one
        channel has raw data and an associated station file. Furthermore it
        must be possible to derive coordinates for the station.

        :type event_name: str
        :param event_name: Name of the event.
        """
        waveform_file = self.comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="raw")

        if list_only:
            with pyasdf.ASDFDataSet(waveform_file, mode="r") as ds:
                return ds.waveforms.list()

        with pyasdf.ASDFDataSet(waveform_file, mode="r") as ds:
            return ds.get_all_coordinates()

    def get_coordinates_for_station(self, event_name, station_id):
        """
        Get the coordinates for one station.

        Must be in sync with :meth:`~.get_all_stations_for_event`.
        """
        waveform_file = self.comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="raw")

        with pyasdf.ASDFDataSet(waveform_file, mode="r") as ds:
            return ds.waveforms[station_id].coordinates

    def get_stations_for_all_events(self):
        """
        Returns a dictionary with a list of stations per event.
        """
        events = {}
        for event in self.comm.events.list():
            try:
                data = self.get_all_stations_for_event(event, list_only=True)
            except LASIFNotFoundError:
                continue
            events[event] = data
        return events

    def get_matching_waveforms(self, event, iteration, station_or_channel_id):
        seed_id = station_or_channel_id.split(".")
        if len(seed_id) == 2:
            channel = None
            station_id = station_or_channel_id
        elif len(seed_id) == 4:
            network, station, _, channel = seed_id
            station_id = ".".join((network, station))
        else:
            raise ValueError("'station_or_channel_id' must either have "
                             "2 or 4 parts.")

        iteration_long_name = self.comm.iterations.get_long_iteration_name(
            iteration)
        event = self.comm.events.get(event)

        # Get the metadata for the processed and synthetics for this
        # particular station.
        data = self.comm.waveforms.get_waveforms_processed(
            event["event_name"], station_id,
            tag=self.comm.waveforms.preprocessing_tag)
        # data_fly = self.comm.waveforms.get_waveforms_processed_on_the_fly(
        #     event["event_name"], station_id)

        synthetics = self.comm.waveforms.get_waveforms_synthetic(
            event["event_name"], station_id,
            long_iteration_name=iteration_long_name)
        coordinates = self.comm.query.get_coordinates_for_station(
            event["event_name"], station_id)

        # Clear data and synthetics!
        for _st, name in ((data, "observed"), (synthetics, "synthetic")):
            # Get all components and loop over all components.
            _comps = set(tr.stats.channel[-1].upper() for tr in _st)
            for _c in _comps:
                traces = [_i for _i in _st
                          if _i.stats.channel[-1].upper() == _c]
                if len(traces) == 1:
                    continue
                elif len(traces) > 1:
                    traces = sorted(traces, key=lambda x: x.id)
                    warnings.warn(
                        "%s data for event '%s', iteration '%s', "
                        "station '%s', and component '%s' has %i traces: "
                        "%s. LASIF will select the first one, but please "
                        "clean up your data." % (
                            name.capitalize(), event["event_name"],
                            iteration, station_id, _c,
                            len(traces), ", ".join(tr.id for tr in traces)),
                        LASIFWarning)
                    for tr in traces[1:]:
                        _st.remove(tr)
                else:
                    # Should not happen.
                    raise NotImplementedError

        # Make sure all data has the corresponding synthetics. It should not
        # happen that one has three channels of data but only two channels
        # of synthetics...in that case, discard the additional data and
        # raise a warning.
        temp_data = []
        for data_tr in data:
            component = data_tr.stats.channel[-1].upper()
            synthetic_tr = [tr for tr in synthetics
                            if tr.stats.channel[-1].upper() == component]
            if not synthetic_tr:
                warnings.warn(
                    "Station '%s' has observed data for component '%s' but no "
                    "matching synthetics." % (station_id, component),
                    LASIFWarning)
                continue
            temp_data.append(data_tr)
        data.traces = temp_data

        if len(data) == 0:
            raise LASIFError("No data remaining for station '%s'." %
                             station_id)

        # Scale the data if required.
        if self.comm.project.processing_params["scale_data_to_synthetics"]:
            for data_tr in data:
                synthetic_tr = [
                    tr for tr in synthetics
                    if tr.stats.channel[-1].lower() ==
                    data_tr.stats.channel[-1].lower()][0]
                scaling_factor = synthetic_tr.data.ptp() / \
                    data_tr.data.ptp()
                # Store and apply the scaling.
                data_tr.stats.scaling_factor = scaling_factor
                data_tr.data *= scaling_factor

        data.sort()
        synthetics.sort()

        # Select component if necessary.
        if channel and channel is not None:
            # Only use the last letter of the channel for the selection.
            # Different solvers have different conventions for the location
            # and channel codes.
            component = channel[-1].upper()
            data.traces = [i for i in data.traces
                           if i.stats.channel[-1].upper() == component]
            synthetics.traces = [i for i in synthetics.traces
                                 if i.stats.channel[-1].upper() == component]

        return DataTuple(data=data, synthetics=synthetics,
                         coordinates=coordinates)

    def point_in_domain(self, latitude, longitude):
        """
        Tests if the point is in the domain. Returns True/False

        :param latitude: The latitude of the point.
        :param longitude: The longitude of the point.
        """
        domain = self.comm.project.domain
        return domain.point_in_domain(longitude=longitude, latitude=latitude)
