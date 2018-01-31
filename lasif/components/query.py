#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import os
import warnings

from lasif import LASIFError, LASIFNotFoundError, LASIFWarning

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
    def get_all_stations_for_event(self, event_name):
        """
        Returns a list of all stations for one event.

        A station is considered to be available for an event if at least one
        channel has raw data and an associated station file. Furthermore it
        must be possible to derive coordinates for the station.

        :type event_name: str
        :param event_name: Name of the event.
        """

        import pyasdf

        waveform_file = self.comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="raw")

        with pyasdf.ASDFDataSet(waveform_file, mode="r") as ds:
            return ds.get_all_coordinates()

    def get_coordinates_for_station(self, event_name, station_id):
        """
        Get the coordinates for one station.

        Must be in sync with :meth:`~.get_all_stations_for_event`.
        """
        import pyasdf

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
                data = self.get_all_stations_for_event(event).keys()
            except LASIFNotFoundError:
                continue
            events[event] = data
        return events

    def get_iteration_status(self, iteration, events=None):
        """
        Return the status of an iteration. This is a query command as it
        integrates a lot of different information.

        :param iteration: The iteration which to query.
        :param events: If given, only the events of those events will be
            queried. Otherwise all will be queried.

        Returns a dictionary of events, each containing the following keys:
        ``"missing_raw"``, ``"missing_processed"``,
        ``"missing_synthetic"``, ``"fraction_of_stations_that_have_windows"``

        Each of those is a list of stations missing for that particular data
        type.
        """
        iteration = self.comm.iterations.get(iteration)

        # Collect the status information per event.
        status = collections.defaultdict(dict)

        # Make sure events is a list of event names to ease the later check.
        if events:
            events = [self.comm.events.get(_i)["event_name"] for _i in events]

        # Get all the data.
        for event_name, event_dict in iteration.events.items():
            # Skip events if some are specified.
            if events and event_name not in events:
                    continue

            # Get all stations that should be defined for the current
            # iteration.
            stations = set(event_dict["stations"].keys())

            # Raw data.
            try:
                raw = self.comm.waveforms.get_metadata_raw(event_name)
                # Get a list of all stations
                raw = set((
                    "{network}.{station}".format(**_i) for _i in raw))
                # Get the missing raw stations.
                missing_raw = stations.difference(raw)
            except LASIFNotFoundError:
                missing_raw = set(stations)
            status[event_name]["missing_raw"] = missing_raw

            # Processed data.
            try:
                processed = self.comm.waveforms.get_metadata_processed(
                    event_name, iteration.processing_tag)
                # Get a list of all stations
                processed = set((
                    "{network}.{station}".format(**_i) for _i in processed))
                # Get all stations in raw that are also defined for the
                # current iteration.
                # Get the missing raw stations.
                missing_processed = stations.difference(processed)
            except LASIFNotFoundError:
                missing_processed = set(stations)
            status[event_name]["missing_processed"] = missing_processed

            # Synthetic data.
            try:
                synthetic = self.comm.waveforms.get_metadata_synthetic(
                    event_name, iteration.long_name)
                # Get a list of all stations
                synthetic = set((
                    "{network}.{station}".format(**_i) for _i in synthetic))
                # Get all stations in raw that are also defined for the
                # current iteration.
                missing_synthetic = stations.difference(synthetic)
            except LASIFNotFoundError:
                missing_synthetic = set(stations)
            status[event_name]["missing_synthetic"] = missing_synthetic

            try:
                windows = self.comm.windows.get(event_name, iteration)
            except LASIFNotFoundError:
                windows = 0
            if windows:
                # Get the windows per station.
                windows = set(".".join(_i.split(".")[:2]) for _i in
                              windows.list())
                windows = stations.intersection(windows)
            status[event_name]["fraction_of_stations_that_have_windows"] = \
                float(len(windows)) / float(len(stations))

        return dict(status)

    def get_data_and_synthetics_iterator(self, iteration, event):
        """
        Get the processed data and matching synthetics for a particular event.
        """
        from ..tools.data_synthetics_iterator import DataSyntheticIterator
        return DataSyntheticIterator(self.comm, iteration, event)

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

        for tag in proc_tags:
            try:
                procs = self.comm.waveforms.get_metadata_processed_for_station(
                    event_name, tag, station_id)
            except LASIFNotFoundError:
                continue
            comps = get_components(procs)
            if not comps:
                continue
            all_files["processed"][tag] = comps

        iterations = self.comm.waveforms.get_available_synthetics(event_name)
        synthetic_coordinates_mapping = {"X": "N",
                                         "Y": "E",
                                         "Z": "Z",
                                         "N": "N",
                                         "E": "E"}
        for it in iterations:
            try:
                its = self.comm.waveforms.get_metadata_synthetic_for_station(
                    event_name, it, station_id)
            except LASIFNotFoundError:
                continue
            comps = get_components(its)
            if not comps:
                continue
            comps = [synthetic_coordinates_mapping[i] for i in comps]
            all_files["synthetic"][it] = sorted(comps, reverse=True)
        return all_files

    def point_in_domain(self, latitude, longitude):
        """
        Tests if the point is in the domain. Returns True/False

        :param latitude: The latitude of the point.
        :param longitude: The longitude of the point.
        """
        domain = self.comm.project.domain
        return domain.point_in_domain(longitude=longitude, latitude=latitude)

    def what_is(self, path):
        """
        Debug function returning a string with information about the file.
        Useful as a debug function and to figure out what LASIF is doing.

        :param path: The path to the file.
        """
        path = os.path.normpath(os.path.abspath(path))

        # File does not exist.
        if not os.path.exists(path):
            raise LASIFNotFoundError("Path '%s' does not exist." % path)
        # File not part of the project.
        if os.path.commonprefix([path, self.comm.project.paths["root"]]) \
                != self.comm.project.paths["root"]:
            raise LASIFError("File '%s' is not part of the LASIF project." %
                             path)

        # Split in dir an folder to ease the rest.
        if os.path.isdir(path):
            return self.__what_is_this_folder(path)
        else:
            return self.__what_is_this_file(path)

    def __what_is_this_folder(self, folder_path):
        key = [_i[0] for _i in self.comm.project.paths.items() if _i[1] ==
               folder_path]
        if key:
            key = key[0]
            info = {
                "kernels": "Folder storing all kernels and gradients.",
                "synthetics": "Folder storing synthetic waveforms.",
                "config_file": "The configuration file.",
                "logs": "Folder storing logs of various operations.",
                "config_file_cache": "The cache file for the config file.",
                "stations": "Folder storing all station files.",
                "models": "Folder storing Earth models.",
                "root": "The root directory of the project.",
                "resp": "Folder storing station files in the resp format.",
                "cache": "Folder storing various intermediate caches.",
                "inv_db_file": "Database for coordinates from the internet.",
                "station_xml": "Folder storing StationXML station files.",
                "adjoint_sources": "Folder storing adjoint sources.",
                "windows": "Folder storing window definition files.",
                "wavefields": "Folder storing wavefields.",
                "dataless_seed": "Folder storing dataless SEED station files.",
                "iterations": "Folder storing the iteration definitions.",
                "output": "Folder storing the output of various operations.",
                "data": "Folder storing raw and processed waveforms.",
                "events": "Folder storing event definitions as QuakeML."
            }

            if key in info:
                ret_str = info[key]
            else:
                ret_str = "Project folder '%s'." % key
            return ret_str
        else:
            return None

    def __what_is_this_file(self, file_path):
        key = [_i[0] for _i in self.comm.project.paths.items() if _i[1] ==
               file_path]
        # Deal with files defined by the project itsself.
        if key:
            key = key[0]
            info = {
                "config_file": "The main project configuration file.",
                "config_file_cache": "The cache file for the config file.",
                "inv_db_file": "Database for coordinates from the internet.",
            }

            if key in info:
                ret_str = info[key]
            else:
                ret_str = "Project file '%s'." % key
            return ret_str
        # Deal with other files.
        else:
            # Check if it is a subfolder of any of the other defined paths.
            common_prefix = [_i for _i in self.comm.project.paths.items() if
                             os.path.commonprefix([_i[1], file_path]) == _i[1]]
            # Not a project file if nothing is found.
            if not common_prefix:
                return None
            common_prefix.sort(key=lambda x: len(x[1]))
            common_prefix = common_prefix[-1][0]
            if common_prefix in ["dataless_seed", "resp", "stationxml"]:
                return self.__what_is_this_station_file(file_path,
                                                        filetype=common_prefix)
            elif common_prefix in ["data", "synthetics"]:
                return self.__what_is_this_waveform_file(
                    file_path, filetype=common_prefix)
            else:
                raise NotImplementedError
            return None

    def __what_is_this_waveform_file(self, file_path, filetype):
        import obspy
        info = self.comm.waveforms.get_metadata_for_file(file_path)
        st = obspy.read(file_path)

        info = [
            "\t%s | %s - %s | Lat/Lng/Ele/Dep: %s/%s/%s/%s" % (
                _i["channel_id"],
                str(obspy.UTCDateTime(_i["starttime_timestamp"])),
                str(obspy.UTCDateTime(_i["endtime_timestamp"])),
                "%.2f" % _i["latitude"]
                if _i["latitude"] is not None else "--",
                "%.2f" % _i["longitude"]
                if _i["longitude"] is not None else "--",
                "%.2f" % _i["elevation_in_m"]
                if _i["elevation_in_m"] is not None else "--",
                "%.2f" % _i["local_depth_in_m"]
                if _i["local_depth_in_m"] is not None else "--")
            for _i in info]

        return (
            "The %s file contains information about %i channel%s:\n%s" % (
                st[0].stats._format, len(info),
                "s" if len(info) > 1 else "",
                "\n".join(info)))

    def __what_is_this_station_file(self, file_path, filetype):
        ft_map = {
            "dataless_seed": "SEED", "resp": "RESP", "stationxml": "StationXML"
        }
        info = self.comm.stations.get_details_for_filename(file_path)
        if info is None:
            raise LASIFNotFoundError("File '%s' is not valid station file." %
                                     file_path)

        info = [
            "\t%s | %s - %s | Lat/Lng/Ele/Dep: %s/%s/%s/%s" % (
                _i["channel_id"],
                str(_i["start_date"]),
                str(_i["end_date"]) if _i["end_date"] else "--",
                "%.2f" % _i["latitude"]
                if _i["latitude"] is not None else "--",
                "%.2f" % _i["longitude"]
                if _i["longitude"] is not None else "--",
                "%.2f" % _i["elevation_in_m"]
                if _i["elevation_in_m"] is not None else "--",
                "%.2f" % _i["local_depth_in_m"]
                if _i["local_depth_in_m"] is not None else "--")
            for _i in info]

        return (
            "The %s file contains information about %i channel%s:\n%s" % (
                ft_map[filetype], len(info), "s" if len(info) > 1 else "",
                "\n".join(info)))
