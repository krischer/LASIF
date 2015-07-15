#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

import obspy

from lasif import LASIFNotFoundError
from .component import Component


class StationsComponent(Component):
    """
    Component responsible for dealing with station information in either
    StationXML, SEED, or RESP formats. The information is always bound to a
    specific channel and time and never to the whole station despite the
    name of this component. Always use this component to get the required
    station related information
    and do not do it yourself.

    StationXML files must adhere to the naming scheme ``*.xml``, SEED files
    to ``dataless.*``, and RESP files to ``RESP.*``. They must be stored in
    separate, non-nested folders.

    The term ``channel_id`` always means the full SEED identifier,
    e.g. network, station , location, and channel codes.

    :param stationxml_folder: The StationXML folder.
    :param seed_folder: The dataless SEED folder.
    :param resp_folder: The RESP files folder.
    :param cache_folder: The folder where the cache file should be
        stored. The file will be named ``station_cache.sqlite``.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the
        communicator.
    """
    def __init__(self, stationxml_folder, seed_folder, resp_folder,
                 cache_folder, communicator, component_name):
        self.cache_folder = cache_folder
        self.stationxml_folder = stationxml_folder
        self.seed_folder = seed_folder
        self.resp_folder = resp_folder

        # Attribute will store the station cache if it has been initialized.
        self.__cached_station_cache = None
        self.cache_file = os.path.join(self.cache_folder,
                                       "station_cache.sqlite")

        super(StationsComponent, self).__init__(communicator, component_name)

    @property
    def _station_cache(self):
        """
        Cached access to the station cache.
        """
        if self.__cached_station_cache is not None:
            return self.__cached_station_cache
        else:
            return self.force_cache_update()

    @property
    def file_count(self):
        """
        Returns the total number of station files.
        """
        return self._station_cache.file_count

    @property
    def total_file_size(self):
        """
        Returns the filesize sum of all station files.
        """
        return self._station_cache.total_size

    def force_cache_update(self):
        """
        Update the station cache.

        The station cache is used to cache the information on stations so
        LASIF does not constantly open files which would be very slow. This
        method is only necessary if you change/add/remove a station file and
        need it to be reflected immediatly. The next invocation of LASIF
        will update the cache automatically.
        """
        from ..tools.cache_helpers.station_cache import StationCache
        self.__cached_station_cache = StationCache(
            self.cache_file,
            root_folder=self.comm.project.paths["root"],
            seed_folder=self.seed_folder,
            resp_folder=self.resp_folder,
            stationxml_folder=self.stationxml_folder,
            read_only=self.comm.project.read_only_caches)
        return self.__cached_station_cache

    def get_details_for_filename(self, filename):
        """
        Returns the details for a single file. Each file can have more than
        channel stored in it. Returns a list of dictionaries.

        :param filename: The name of the station file.
        :type filename: str

        >>> comm = getfixture('stations_comm')
        >>> filename = sorted(comm.stations.get_all_channels())[1]["filename"]
        >>> comm.stations.get_details_for_filename(filename) \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [{'local_depth_in_m': 0.0, 'end_date': None, 'elevation_in_m': 565.0,
          'longitude': 11.2752, 'filename':  u'/.../dataless.BW_FURT',
          'channel_id': u'BW.FURT..EHZ', 'latitude': 48.162899,
          'start_date': UTCDateTime(2001, 1, 1, ...}, ...]

        The values for ``end_date`` and ``local_depth_in_m`` can be ``None``.
        """
        values = self._station_cache.get_details(filename)
        for value in values:
            value["start_date"] = obspy.UTCDateTime(value["start_date"])
            if not value["end_date"]:
                continue
            value["end_date"] = obspy.UTCDateTime(value["end_date"])
        return values

    def get_all_channels(self):
        """
        Get information about all available channels at all points in time.

        Returns a list of dictionaries.

        Mainly useful for testing and debugging. For real applications use
        :meth:`.get_all_channels_at_time` which will return all channels
        active at a certain point in time.

        >>> comm = getfixture('stations_comm')
        >>> import pprint
        >>> pprint.pprint(sorted(comm.stations.get_all_channels(),
        ...                      key=lambda x: x["channel_id"], reverse=True))\
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [{'channel_id': u'IU.PAB.00.BHE',
          'elevation_in_m': 950.0,
          'end_date': UTCDateTime(2009, 8, 13, 19, 0),
          'filename': u'.../station_files/seed/dataless.IU_PAB',
          'latitude': 39.5446,
          'local_depth_in_m': 0.0,
          'longitude': -4.349899,
          'start_date': UTCDateTime(1999, 2, 18, 10, 0)}, {...}, ...]

        Returns a list of dictionaries, each with the following keys:

        >>> sorted(comm.stations.get_all_channels()[0].keys()) \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ['channel_id', 'elevation_in_m', 'end_date', 'filename', 'latitude',
         'local_depth_in_m', 'longitude', 'start_date']

        The values for ``end_date`` and ``local_depth_in_m`` can be ``None``.
        """
        values = self._station_cache.get_values()
        for value in values:
            value["start_date"] = obspy.UTCDateTime(value["start_date"])
            if not value["end_date"]:
                continue
            value["end_date"] = obspy.UTCDateTime(value["end_date"])
        return values

    def get_all_channels_at_time(self, time):
        """
        Returns a dictionary with coordinates for all channels available at
        a certain point in time.

        :param time: The time or timestamp at which to return available
            channels.
        :type time: ``int`` or :class:`~obspy.core.utcdatetime.UTCDateTime`
        :returns: All available coordinates. The keys are the channel ids
            and the values are dictionaries with the coordinates.

        >>> from obspy import UTCDateTime
        >>> comm = getfixture('stations_comm')
        >>> comm.stations.get_all_channels_at_time(UTCDateTime(2012, 1, 1)) \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {u'BW.FURT..EHE': {'latitude': 48.162899, 'elevation_in_m': 565.0,
                           'local_depth_in_m': 0.0, 'longitude': 11.2752},
         ...}

        It also works with timestamps.

        >>> comm.stations.get_all_channels_at_time(1325376000) \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {u'BW.FURT..EHE': {'latitude': 48.162899, 'elevation_in_m': 565.0,
                           'local_depth_in_m': 0.0, 'longitude': 11.2752},
         ...}

        The value for ``local_depth_in_m`` can be ``None``.
        """
        return self._station_cache.get_all_channels_at_time(time)

    def has_channel(self, channel_id, time):
        """
        Boolean test if information for the given channel and time is
        available.

        :param channel_id: The id of the channel.
        :type channel_id: str
        :param time: The time at which to retrieve the information.
        :type time: ``int`` or :class:`~obspy.core.utcdatetime.UTCDateTime`

        >>> from obspy import UTCDateTime
        >>> comm = getfixture('stations_comm')

        It works with :class:`~obspy.core.utcdatetime.UTCDateTime` objects

        >>> comm.stations.has_channel('IU.ANMO.10.BHZ',
        ...                           UTCDateTime(2012, 3, 14))
        True

        and timestamps.

        >>> comm.stations.has_channel('IU.ANMO.10.BHZ', 1331683200)
        True
        >>> comm.stations.has_channel('AA.BB.CC.DD', 1331683200)
        False
        """
        return self._station_cache.station_info_available(channel_id, time)

    def get_channel_filename(self, channel_id, time):
        """
        Returns the absolute path of the file storing the information for the
        given channel and time combination.

        :param channel_id: The id of the channel.
        :type channel_id: str
        :param time: The time at which to retrieve the information.
        :type time: ``int`` or :class:`~obspy.core.utcdatetime.UTCDateTime`

        >>> import obspy
        >>> comm = getfixture('stations_comm')

        It works with :class:`~obspy.core.utcdatetime.UTCDateTime` objects

        >>> comm.stations.get_channel_filename(  # doctest: +ELLIPSIS
        ...     "IU.ANMO.10.BHZ", obspy.UTCDateTime(2012, 3, 14))
        u'/.../IRIS_single_channel_with_response.xml'

        and timestamps.

        >>> comm.stations.get_channel_filename(  # doctest: +ELLIPSIS
        ...     "IU.ANMO.10.BHZ", 1331683200)
        u'/.../IRIS_single_channel_with_response.xml'
        """
        filename = self._station_cache.get_station_filename(channel_id, time)
        if filename is None:
            raise LASIFNotFoundError(
                "Could not find a station file for channel '%s' at %s." % (
                    channel_id, str(time)))
        return filename

    def get_station_filename(self, network, station, location, channel,
                             file_format):
        """
        Function returning the filename a station file of a certain format
        should be written to. Only useful as a callback function for
        downloaders or other modules saving data to LASIF.

        :type file_format: str
        :param file_format: 'datalessSEED', 'StationXML', or 'RESP'

        >>> comm = getfixture('stations_comm')
        >>> comm.stations.get_station_filename(
        ...     "BW", "FURT", "", "BHZ", file_format="RESP")\
        # doctest: +ELLIPSIS
        '/.../resp/RESP.BW.FURT..BHZ'

        >>> comm.stations.get_station_filename(
        ...     "BW", "ALTM", "", "BHZ", file_format="datalessSEED")\
        # doctest: +ELLIPSIS
        '/.../seed/dataless.BW_ALTM'

        >>> comm.stations.get_station_filename(
        ...     "BW", "FURT", "", "BHZ", file_format="StationXML")\
        # doctest: +ELLIPSIS
        '/.../stationxml/BW_FURT.xml'
        """
        if file_format not in ["datalessSEED", "StationXML", "RESP"]:
            msg = "Unknown format '%s'" % file_format
            raise ValueError(msg)
        if file_format == "datalessSEED":
            def seed_filename_generator():
                i = 0
                while True:
                    filename = os.path.join(
                        self.seed_folder,
                        "dataless.{network}_{station}".format(
                            network=network, station=station))
                    if i:
                        filename += ".%i" % i
                    i += 1
                    yield filename
            for filename in seed_filename_generator():
                if not os.path.exists(filename):
                    break
            return filename
        elif file_format == "RESP":
            def resp_filename_generator():
                i = 0
                while True:
                    filename = os.path.join(
                        self.resp_folder,
                        "RESP.{network}.{station}.{location}.{channel}"
                        .format(network=network, station=station,
                                location=location, channel=channel))
                    if i:
                        filename += ".%i" % i
                    i += 1
                    yield filename
            for filename in resp_filename_generator():
                if not os.path.exists(filename):
                    break
            return filename
        elif file_format == "StationXML":
            def stationxml_filename_generator():
                i = 0
                while True:
                    filename = os.path.join(
                        self.stationxml_folder,
                        "{network}_{station}{i}.xml".format(
                            network=network, station=station,
                            i=".%i" if i else ""))
                    i += 1
                    yield filename
            for filename in stationxml_filename_generator():
                if not os.path.exists(filename):
                    break
            return filename
        else:
            raise NotImplementedError
