#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import fnmatch
import itertools
import os
import warnings

import obspy

from lasif import LASIFError, LASIFNotFoundError, LASIFWarning
from ..tools.cache_helpers.waveform_cache import WaveformCache
from .component import Component


class LimitedSizeDict(collections.OrderedDict):
    """
    Based on http://stackoverflow.com/a/2437645/1657047
    """
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        collections.OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        collections.OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


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
        # Limit to 20 instances as SQLite does not like too many open
        # databases at the same time.
        self.__cache = LimitedSizeDict(size_limit=10)

        super(WaveformsComponent, self).__init__(communicator, component_name)

    def reset_cached_caches(self):
        """
        The waveform component caches the actual waveform caches to make its
        access instant. This is usually ok, but sometimes new data is added
        while the same instance of LASIF is still active. Thus the cached
        caches need to be reset at times.
        """
        self.__cache = {}

    def get_metadata_for_file(self, absolute_filename):
        """
        Returns the metadata for a certain file.

        :param absolute_filename: The absolute path of the file.
        """
        if os.path.commonprefix([absolute_filename, self._data_folder]) == \
                self._data_folder:
            relpath = os.path.relpath(absolute_filename, self._data_folder)
            event, type_or_tag, filename = relpath.split(os.path.sep)
            if type_or_tag == "raw":
                c = self.get_waveform_cache(event, "raw")
            else:
                c = self.get_waveform_cache(event, "processed",
                                            type_or_tag)
        elif os.path.commonprefix([absolute_filename, self._data_folder]) == \
                self._synthetics_folder:
            relpath = os.path.relpath(absolute_filename,
                                      self._synthetics_folder)
            event, iteration, filename = relpath.split(os.path.sep)
            c = self.get_waveform_cache(event, "synthetic", iteration)
        else:
            raise LASIFError("Invalid path.")

        return c.get_details(absolute_filename)

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

    def get_waveform_cache(self, event_name, data_type,
                           tag_or_iteration=None, dont_update=False):
        """
        :param event_name: The name of the event.
        :param data_type: The data type.
        :param tag_or_iteration: If processed data, the tag, if synthetic
            data, the iteration.
        :param dont_update: If True, an existing cache will not be updated
            but returned as is. If it does not exist, it will be updated
            regardless.
        """
        if data_type == "synthetic":
            tag_or_iteration = \
                self.comm.iterations.get(tag_or_iteration).long_name

        data_path = self.get_waveform_folder(event_name, data_type,
                                             tag_or_iteration)
        if data_type == "raw":
            if not os.path.exists(data_path):
                msg = "No data for event '%s' found." % event_name
                raise LASIFNotFoundError(msg)
            label = "Raw"
        elif data_type == "processed":
            if not tag_or_iteration:
                msg = "Tag must be given for processed data."
                raise ValueError(msg)
            if not os.path.exists(data_path):
                msg = ("No data for event '%s' and processing tag '%s' "
                       "found." % (event_name, tag_or_iteration))
                raise LASIFNotFoundError(msg)
            label = "Processed %s" % event_name
        elif data_type == "synthetic":
            if not tag_or_iteration:
                msg = "Long iteration name must be given for synthetic data."
                raise ValueError(msg)
            if not os.path.exists(data_path):
                msg = ("No synthetic data for event '%s' and iteration '%s' "
                       "found." % (event_name, tag_or_iteration))
                raise LASIFNotFoundError(msg)
            label = "Synthetic %s" % event_name
        else:
            raise ValueError("Invalid data type '%s'." % data_type)

        waveform_db_file = data_path + "_cache" + os.path.extsep + "sqlite"
        if waveform_db_file in self.__cache:
            return self.__cache[waveform_db_file]
        if dont_update is True and os.path.exists(waveform_db_file):
            cache = WaveformCache(cache_db_file=waveform_db_file,
                                  root_folder=self.comm.project.paths["root"],
                                  waveform_folder=data_path,
                                  pretty_name="%s Waveform Cache" % label,
                                  read_only=True)
        elif data_type == "synthetic" \
                and not os.path.exists(waveform_db_file) \
                and os.listdir(data_path):
            # If it is synthetic, read a file and assume all other files
            # have the same length. This has the huge advantage that the
            # files no longer have to be opened but only the filename has to
            # be parsed. Only works for SES3D files.
            files = sorted(os.listdir(data_path))
            filename = os.path.join(data_path, files[len(files) // 2])
            tr = obspy.read(filename)[0]
            synthetic_info = {
                "starttime_timestamp": tr.stats.starttime.timestamp,
                "endtime_timestamp": tr.stats.endtime.timestamp
            }

            cache = WaveformCache(cache_db_file=waveform_db_file,
                                  root_folder=self.comm.project.paths["root"],
                                  waveform_folder=data_path,
                                  pretty_name="%s Waveform Cache" % label,
                                  read_only=self.comm.project.read_only_caches,
                                  synthetic_info=synthetic_info)
        else:
            cache = WaveformCache(cache_db_file=waveform_db_file,
                                  root_folder=self.comm.project.paths["root"],
                                  waveform_folder=data_path,
                                  pretty_name="%s Waveform Cache" % label,
                                  read_only=self.comm.project.read_only_caches)
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
        :param station_id: The id of the station in the form ``NET.STA``.
        """
        return self._get_waveforms(event_name, station_id, data_type="raw")

    def get_waveforms_processed(self, event_name, station_id, tag):
        """
        Gets the processed waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form ``NET.STA``.
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
        :param station_id: The id of the station in the form ``NET.STA``.
        :param long_iteration_name: The long form of an iteration name.
        """
        from lasif import rotations
        import lasif.domain

        iteration = self.comm.iterations.get(long_iteration_name)

        st = self._get_waveforms(event_name, station_id,
                                 data_type="synthetic",
                                 tag_or_iteration=iteration.long_name)
        network, station = station_id.split(".")

        formats = list(set([tr.stats._format for tr in st]))
        if len(formats) != 1:
            raise ValueError(
                "The synthetics for one Earthquake must all have the same "
                "data format under the assumption that they all originate "
                "from the same solver. Found formats: %s" % (str(formats)))
        format = formats[0].lower()

        # In the case of data coming from SES3D the components must be
        # mapped to ZNE as it works in XYZ.
        if format == "ses3d":
            # This maps the synthetic channels to ZNE.
            synthetic_coordinates_mapping = {"X": "N", "Y": "E", "Z": "Z"}

            for tr in st:
                tr.stats.network = network
                tr.stats.station = station
                # SES3D X points south. Reverse it to arrive at ZNE.
                if tr.stats.channel in ["X"]:
                    tr.data *= -1.0
                # SES3D files have no starttime. Set to the event time.
                tr.stats.starttime = \
                    self.comm.events.get(event_name)["origin_time"]
                tr.stats.channel = \
                    synthetic_coordinates_mapping[tr.stats.channel]

            # Rotate if needed. Again only SES3D synthetics need to be rotated.
            domain = self.comm.project.domain
            if isinstance(domain, lasif.domain.RectangularSphericalSection) \
                    and domain.rotation_angle_in_degree and \
                    "ses3d" in iteration.solver_settings["solver"].lower():
                # Coordinates are required for the rotation.
                coordinates = self.comm.query.get_coordinates_for_station(
                    event_name, station_id)

                # First rotate the station back to see, where it was
                # recorded.
                lat, lng = rotations.rotate_lat_lon(
                    lat=coordinates["latitude"], lon=coordinates["longitude"],
                    rotation_axis=domain.rotation_axis,
                    angle=-domain.rotation_angle_in_degree)
                # Rotate the synthetics.
                n, e, z = rotations.rotate_data(
                    st.select(channel="N")[0].data,
                    st.select(channel="E")[0].data,
                    st.select(channel="Z")[0].data,
                    lat, lng,
                    domain.rotation_axis,
                    domain.rotation_angle_in_degree)
                st.select(channel="N")[0].data = n
                st.select(channel="E")[0].data = e
                st.select(channel="Z")[0].data = z

        st.sort()

        # Apply the project function that modifies synthetics on the fly.
        fct = self.comm.project.get_project_function("process_synthetics")
        return fct(st, iteration=iteration,
                   event=self.comm.events.get(event_name))

    def _get_waveforms(self, event_name, station_id, data_type,
                       tag_or_iteration=None):
        waveform_cache = self.get_waveform_cache(event_name,
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
                warnings.warn(msg, LASIFWarning)
            files = locations[keys[0]]
        st = obspy.Stream()
        for single_file in files:
            st += obspy.read(single_file["filename"])
        st.sort()
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
        >>> pprint.pprint(sorted(comm.waveforms.get_metadata_raw(
        ...     "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"),
        ...     key=lambda x: x["channel_id"])) \
        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [{'channel': u'BHE',
          'channel_id': u'HL.ARG..BHE',
          'elevation_in_m': None,
          'endtime': UTCDateTime(2010, 3, 24, 15, 11, 30, 974999),
          'filename': u'/.../HL.ARG..BHE.mseed',
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
        waveform_cache = self.get_waveform_cache(event_name,
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
        waveform_cache = self.get_waveform_cache(event_name,
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
        waveform_cache = self.get_waveform_cache(event_name,
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
        :param station_id: The id of the station in the form ``NET.STA``.
        """
        waveform_cache = self.get_waveform_cache(event_name,
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
        :param station_id: The id of the station in the form ``NET.STA``.
        """
        # Assure the iteration actually contains the event.
        it = self.comm.iterations.get(long_iteration_name)
        if event_name not in it.events:
            raise LASIFNotFoundError(
                "Iteration '%s' does not contain event '%s'." % (it.name,
                                                                 event_name))
        waveform_cache = self.get_waveform_cache(
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
        :param station_id: The id of the station in the form ``NET.STA``.
        """
        waveform_cache = self.get_waveform_cache(
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

        # Make sure the iterations also contain the event and the stations.
        its = []
        for iteration in iterations:
            try:
                it = self.comm.iterations.get(iteration)
            except LASIFNotFoundError:
                continue
            if event_name not in it.events:
                continue
            its.append(it.name)
        return its
