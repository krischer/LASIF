#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import obspy
import os

from lasif import LASIFNotFoundError
from lasif.tools.waveform_cache import WaveformCache

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

    def _get_waveform_cache_file(self, waveform_db_file, data_path):
        """
        Helper function returning a waveform cache instance.
        """
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
        data_folder = os.path.join(self._data_folder, event_name)
        waveform_db_file = os.path.join(data_folder,
                               "raw_cache" + os.path.extsep + "sqlite")
        data_path = os.path.join(data_folder, "raw")
        if not os.path.exists(data_path):
            msg = "No data for event '%s' found." % event_name
            raise LASIFNotFoundError(msg)
        waveform_cache = self._get_waveform_cache_file(waveform_db_file,
                                                       data_path)
        values = waveform_cache.get_values()
        if not values:
            msg = "No data for event '%s' found." % event_name
            raise LASIFNotFoundError(msg)
        return self._convert_timestamps(waveform_cache.get_values())

    def get_metadata_processed(self, event_name, tag):
        data_folder = os.path.join(self._data_folder, event_name)
        waveform_db_file = os.path.join(
            data_folder, tag + "_cache" + os.path.extsep + "sqlite")
        data_path = os.path.join(data_folder, tag)
        if not os.path.exists(data_path):
            msg = "No data for event '%s' and processing tag '%s' found." % \
                  (event_name, tag)
            raise LASIFNotFoundError(msg)
        waveform_cache = self._get_waveform_cache_file(waveform_db_file,
                                                       data_path)
        values = waveform_cache.get_values()
        if not values:
            msg = "No data for event '%s' and processing tag '%s' found." % \
                  (event_name, tag)
            raise LASIFNotFoundError(msg)
        return self._convert_timestamps(waveform_cache.get_values())

    def get_metadata_synthetic(self, event_name, long_iteration_name):
        data_folder = os.path.join(self._synthetics_folder, event_name)
        waveform_db_file = os.path.join(
            data_folder, long_iteration_name + "_cache" + os.path.extsep +
            "sqlite")
        data_path = os.path.join(data_folder, long_iteration_name)
        if not os.path.exists(data_path):
            msg = ("No synthetic data for event '%s' and iteration '%s' "
                   "found." %
                   (event_name, long_iteration_name))
            raise LASIFNotFoundError(msg)
        waveform_cache = self._get_waveform_cache_file(waveform_db_file,
                                                       data_path)
        values = waveform_cache.get_values()
        if not values:
            msg = ("No synthetic data for event '%s' and iteration '%s' "
                   "found." %
                   (event_name, long_iteration_name))
        return self._convert_timestamps(waveform_cache.get_values())
