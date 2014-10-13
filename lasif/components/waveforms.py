#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import fnmatch
import itertools
import os
import warnings

import obspy

from lasif import LASIFNotFoundError, LASIFWarning
from ..tools.cache_helpers.waveform_cache import WaveformCache
from .component import Component


class WaveformsComponent(Component):
    """
    Component managing the waveform data.

    :param data_folder: The data folder in a LASIF project.
    :param synthetics_folder: The synthetics folder in a LASIF project.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, data_folder, synthetics_folder, communicator,
                 component_name):
        self._data_folder = data_folder
        self._synthetics_folder = synthetics_folder

        # Internal cache for the initialized waveform cache instances.
        self.__cache = {}

        super(WaveformsComponent, self).__init__(communicator, component_name)

    def get_waveform_folder(self, event_name, data_type,
                            tag_or_iteration=None):
        """
        Returns the folder where the waveforms are stored.

        :param event_name: Name of the event.
        :param data_type: The type of data, one of ``"raw"``,
            ``"processed"``, ``"synthetic"``
        :param tag_or_iteration: The processing tag or iteration name if any.
        """
        if data_type == "raw":
            return os.path.join(self._data_folder, event_name, "raw")
        elif data_type == "processed":
            if not tag_or_iteration:
                msg = "Tag must be given for processed data."
                raise ValueError(msg)
            return os.path.join(self._data_folder, event_name,
                                tag_or_iteration)
        elif data_type == "synthetic":
            if not tag_or_iteration:
                msg = "Long iteration name must be given for synthetic data."
                raise ValueError(msg)
            return os.path.join(self._synthetics_folder, event_name,
                                tag_or_iteration)
        else:
            raise ValueError("Invalid data type '%s'." % data_type)

    def _get_waveform_cache_file(self, event_name, data_type,
                                 tag_or_iteration=None):
        data_path = self.get_waveform_folder(event_name, data_type,
                                             tag_or_iteration)
        if data_type == "raw":
            if not os.path.exists(data_path):
                msg = "No data for event '%s' found." % event_name
                raise LASIFNotFoundError(msg)
        elif data_type == "processed":
            if not tag_or_iteration:
                msg = "Tag must be given for processed data."
                raise ValueError(msg)
            if not os.path.exists(data_path):
                msg = ("No data for event '%s' and processing tag '%s' "
                       "found." % (event_name, tag_or_iteration))
                raise LASIFNotFoundError(msg)
        elif data_type == "synthetic":
            if not tag_or_iteration:
                msg = "Long iteration name must be given for synthetic data."
                raise ValueError(msg)
            if not os.path.exists(data_path):
                msg = ("No synthetic data for event '%s' and iteration '%s' "
                       "found." % (event_name, tag_or_iteration))
                raise LASIFNotFoundError(msg)
        else:
            raise ValueError("Invalid data type '%s'." % data_type)

        waveform_db_file = data_path + "_cache" + os.path.extsep + "sqlite"
        if waveform_db_file in self.__cache:
            return self.__cache[waveform_db_file]
        cache = WaveformCache(cache_db_file=waveform_db_file,
                              waveform_folder=data_path)
        self.__cache[waveform_db_file] = cache
        return cache

    def _convert_timestamps(self, values):
        for value in values:
            value["starttime"] = \
                obspy.UTCDateTime(value["starttime_timestamp"])
            value["endtime"] = \
                obspy.UTCDateTime(value["endtime_timestamp"])
            del value["starttime_timestamp"]
            del value["endtime_timestamp"]
        return values

    def get_waveforms_raw(self, event_name, station_id):
        """
        Gets the raw waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form NET.STA.
        """
        return self._get_waveforms(event_name, station_id, data_type="raw")

    def get_waveforms_processed(self, event_name, station_id, tag):
        """
        Gets the processed waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form NET.STA.
        :param tag: The processing tag.
        """
        return self._get_waveforms(event_name, station_id,
                                   data_type="processed", tag_or_iteration=tag)

    def get_waveforms_synthetic(self, event_name, station_id,
                                long_iteration_name):
        """
        Gets the synthetic waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form NET.STA.
        :param long_iteration_name: The long form of an iteration name.
        """
        from lasif import rotations

        st = self._get_waveforms(event_name, station_id,
                                 data_type="synthetic",
                                 tag_or_iteration=long_iteration_name)
        network, station = station_id.split(".")

        iteration = self.comm.iterations.get(long_iteration_name)

        # This maps the synthetic channels to ZNE.
        synthetic_coordinates_mapping = {"X": "N", "Y": "E", "Z": "Z",
                                         "E": "E", "N": "N"}

        for tr in st:
            tr.stats.network = network
            tr.stats.station = station
            if tr.stats.channel in ["X"]:
                tr.data *= -1.0
            tr.stats.starttime = \
                self.comm.events.get(event_name)["origin_time"]
            tr.stats.channel = \
                synthetic_coordinates_mapping[tr.stats.channel]

        if "specfem" not in iteration.solver_settings["solver"].lower():
            # Also need to be rotated.
            domain = self.comm.project.domain

            # Coordinates are required for the rotation.
            coordinates = self.comm.query.get_coordinates_for_station(
                event_name, station_id)

            # First rotate the station back to see, where it was
            # recorded.
            lat, lng = rotations.rotate_lat_lon(
                coordinates["latitude"], coordinates["longitude"],
                domain["rotation_axis"], -domain["rotation_angle"])
            # Rotate the synthetics.
            n, e, z = rotations.rotate_data(
                st.select(channel="N")[0].data,
                st.select(channel="E")[0].data,
                st.select(channel="Z")[0].data,
                lat, lng,
                domain["rotation_axis"],
                domain["rotation_angle"])
            st.select(channel="N")[0].data = n
            st.select(channel="E")[0].data = e
            st.select(channel="Z")[0].data = z

        return st

    def _get_waveforms(self, event_name, station_id, data_type,
                       tag_or_iteration=None):
        waveform_cache = self._get_waveform_cache_file(event_name,
                                                       data_type,
                                                       tag_or_iteration)
        network, station = station_id.split(".")
        files = waveform_cache.get_files_for_station(network, station)
        if len(files) == 0:
            raise LASIFNotFoundError("No '%s' waveform data found for event "
                                     "'%s' and station '%s'." % (data_type,
                                                                 event_name,
                                                                 station_id))
        if data_type in ["raw", "processed"]:
            # Sort files by location.
            locations = {key: list(value) for key, value in itertools.groupby(
                files, key=lambda x: x["location"])}
            keys = sorted(locations.keys())
            if len(keys) != 1:
                msg = ("Found %s waveform data from %i locations for event "
                       "'%s' and station '%s': %s. Will only use data from "
                       "location '%s'." % (
                           data_type, len(keys), event_name, station_id,
                           ", ".join(["'%s'" % _i for _i in keys]), keys[0]))
                warnings.warn(LASIFWarning, msg)
            files = locations[keys[0]]
        st = obspy.Stream()
        for single_file in files:
            st += obspy.read(single_file["filename"])
        return st

    def get_metadata_raw(self, event_name):
        """
        Returns the available metadata at the channel level for the raw
        waveforms and the given event_name.

        :param event_name: The name of the event.
        :returns: A list of dictionaries, each describing channel level data
            at a particular point in time.

        In most instances, the ``latitude``, ``longitude``,
        ``elevation_in_m``, and ``local_depth_in_m`` values will be
        ``None`` as station information is usually retrieved from the
        station metadata file. SAC files are an exception.

        >>> import pprint
        >>> comm = getfixture('waveforms_comm')
        >>> pprint.pprint(comm.waveforms.get_metadata_raw(
        ...     "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")) \
        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [{'channel': u'BHZ',
          'channel_id': u'HL.ARG..BHZ',
          'elevation_in_m': None,
          'endtime': UTCDateTime(2010, 3, 24, 15, 11, 30, 974999),
          'filename': u'/.../HL.ARG..BHZ.mseed',
          'latitude': None,
          'local_depth_in_m': None,
          'location': u'',
          'longitude': None,
          'network': u'HL',
          'starttime': UTCDateTime(2010, 3, 24, 14, 6, 31, 24999),
          'station': u'ARG'},
         {...},
        ...]

        Each dictionary will have the following keys.

        >>> sorted(comm.waveforms.get_metadata_raw(
        ...     "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")[0].keys()) \
        # doctest: +NORMALIZE_WHITESPACE
        ['channel', 'channel_id', 'elevation_in_m', 'endtime',
         'filename', 'latitude', 'local_depth_in_m', 'location',
         'longitude', 'network', 'starttime', 'station']

        A :class:`~lasif.LASIFNotFoundError` will be raised, if no raw
        waveform data is found for the specified event.

        >>> comm.waveforms.get_metadata_raw("RandomEvent")
        Traceback (most recent call last):
            ...
        LASIFNotFoundError: ...
        """
        waveform_cache = self._get_waveform_cache_file(event_name,
                                                       data_type="raw")
        values = waveform_cache.get_values()
        if not values:
            msg = "No data for event '%s' found." % event_name
            raise LASIFNotFoundError(msg)
        return self._convert_timestamps(waveform_cache.get_values())

    def get_metadata_raw_for_station(self, event_name, station_id):
        """
        Returns the available metadata at the channel level for the raw
        waveforms and the given event_name at a certain station.

        Same as :meth:`~.get_metadata_raw` just for only a single station.

        :param event_name: The name of the event.
        :param station_id: The station id.
        :returns: A list of dictionaries, each describing channel level data
            at a particular point in time.
        """
        waveform_cache = self._get_waveform_cache_file(event_name,
                                                       data_type="raw")
        network_id, station_id = station_id.split(".")
        values = waveform_cache.get_files_for_station(network_id, station_id)
        if not values:
            msg = "No data for event '%s' and station '%s' found." % (
                event_name, station_id)
            raise LASIFNotFoundError(msg)
        return self._convert_timestamps(values)

    def get_metadata_processed(self, event_name, tag):
        """
        Get the processed metadata.

        :param event_name: The name of the event.
        :param tag: The processing tag.
        """
        waveform_cache = self._get_waveform_cache_file(event_name,
                                                       data_type="processed",
                                                       tag_or_iteration=tag)
        values = waveform_cache.get_values()
        if not values:
            msg = "No data for event '%s' and processing tag '%s' found." % \
                  (event_name, tag)
            raise LASIFNotFoundError(msg)
        return self._convert_timestamps(waveform_cache.get_values())

    def get_metadata_processed_for_station(self, event_name, tag, station_id):
        """
        Get the processed metadata for a single station.

        Same as :meth:`~.get_metadata_processed` but for a single station.

        :param event_name: The name of the event.
        :param tag: The processing tag.
        :param station_id: The id of the station in the form NET.STA.
        """
        waveform_cache = self._get_waveform_cache_file(event_name,
                                                       data_type="processed",
                                                       tag_or_iteration=tag)
        network_id, station_id = station_id.split(".")
        values = waveform_cache.get_files_for_station(network_id, station_id)
        if not values:
            msg = "No processed data for event '%s', tag '%s', and station " \
                  "'%s' found." % (event_name, tag, station_id)
            raise LASIFNotFoundError(msg)
        return self._convert_timestamps(values)

    def get_metadata_synthetic(self, event_name, long_iteration_name):
        """
        Get the synthetics metadata.

        :param event_name: The name of the event.
        :param long_iteration_name: The long form of the iteration name.
        :param station_id: The id of the station in the form NET.STA.
        """
        waveform_cache = self._get_waveform_cache_file(
            event_name, data_type="synthetic",
            tag_or_iteration=long_iteration_name)
        values = waveform_cache.get_values()
        if not values:
            msg = ("No synthetic data for event '%s' and iteration '%s' "
                   "found." %
                   (event_name, long_iteration_name))
            raise LASIFNotFoundError(msg)
        return self._convert_timestamps(waveform_cache.get_values())

    def get_metadata_synthetic_for_station(self, event_name,
                                           long_iteration_name, station_id):
        """
        Get the synthetics metadata for a single station.

        Same as :meth:`~.get_metadata_synthetic` but for a single station.

        :param event_name: The name of the event.
        :param long_iteration_name: The long form of the iteration name.
        :param station_id: The id of the station in the form NET.STA.
        """
        waveform_cache = self._get_waveform_cache_file(
            event_name, data_type="synthetic",
            tag_or_iteration=long_iteration_name)
        network_id, station_id = station_id.split(".")
        values = waveform_cache.get_files_for_station(network_id, station_id)
        if not values:
            msg = "No synthetic data for event '%s', iteration '%s', " \
                  "and station '%s' found." % (event_name, long_iteration_name,
                                               station_id)
            raise LASIFNotFoundError(msg)
        return self._convert_timestamps(values)

    def get_available_processing_tags(self, event_name):
        """
        Returns the available processing tags for a given event.

        :param event_name: The event name.
        """
        data_dir = os.path.join(self._data_folder, event_name)
        if not os.path.exists(data_dir):
            raise LASIFNotFoundError("No data for event '%s'." % event_name)
        tags = []
        for tag in os.listdir(data_dir):
            # Only interested in preprocessed data.
            if not tag.startswith("preprocessed") or \
                    tag.endswith("_cache.sqlite"):
                continue
            tags.append(tag)
        return tags

    def get_available_synthetics(self, event_name):
        """
        Returns the available synthetics for a given event.

        :param event_name: The event name.
        """
        data_dir = os.path.join(self._synthetics_folder, event_name)
        if not os.path.exists(data_dir):
            raise LASIFNotFoundError("No synthetic data for event '%s'." %
                                     event_name)
        iterations = []
        for folder in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(
                    self._synthetics_folder, event_name, folder)) \
                    or not fnmatch.fnmatch(folder, "ITERATION_*"):
                continue
            iterations.append(folder)
        return iterations
