from __future__ import absolute_import

import os

from lasif import LASIFNotFoundError
from lasif.tools.station_cache import StationCacheError

from .component import Component


class StationsComponent(Component):
    def __init__(self, stationxml_folder, seed_folder, resp_folder,
                 cache_folder, communicator, component_name):
        self.cache_folder = cache_folder
        self.stationxml_folder = stationxml_folder
        self.seed_folder = seed_folder
        self.resp_folder = resp_folder

        # Attribute will store the station cache if it has been initialized.
        self.__cached_station_cache = None

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

    def force_cache_update(self, show_progress=True):
        """
        Update the station cache.
        """
        from lasif.tools.station_cache import StationCache
        self.__cached_station_cache = StationCache(
            os.path.join(self.cache_folder, "station_cache.sqlite"),
            self.seed_folder, self.resp_folder, self.stationxml_folder,
            show_progress=show_progress)
        return self.__cached_station_cache

    def get_all_channels(self):
        """
        Returns a list of dictionaries, each describing a single channel at
        a certain point in time.

        Mainly useful for testing and debugging.
        """
        return self._station_cache.get_values()

    def has_channel(self, channel_id, time):
        """
        Boolean test if information for the given channel and time is
        available.
        """
        return self._station_cache.station_info_available(channel_id, time)

    def get_station_filename(self, channel_id, time):
        """
        Returns the filename where the information for the given channel and
        time is stored in.
        """
        filename = self._station_cache.get_station_filename(channel_id, time)
        if filename is None:
            raise LASIFNotFoundError
        return filename

    def get_all_coordinates_at_time(self, time):
        """
        Returns a dictionary with coordinates for all channels available at
        a certain point in time.
        """
        return self._station_cache.get_all_channel_coordinates(time)

    def get_coordinates(self, network_code, station_code):
        """
        Get the coordinates for a single station. Will choose the first
        channel it can find for a given stations.
        """
        try:
            coods = self._station_cache.get_coordinates_for_station(
                network_code, station_code)
        except StationCacheError:
            raise LASIFNotFoundError
        return coods

    def get_details_for_filename(self, filename):
        """
        Returns the details for a single file.
        """
        return self._station_cache.get_details(filename)
