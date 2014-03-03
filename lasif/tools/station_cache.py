#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Station Cache class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import glob
from obspy.xseed import Parser
import os
import sqlite3
import warnings

from lasif.tools import simple_resp_parser
from lasif.tools.file_info_cache import FileInfoCache

# The tolerances to which station coordinates for different locations are
# considered equal.
TOL_DEGREES = 0.01
TOL_METERS = 1000.0


class StationCache(FileInfoCache):
    """
    Cache for Station files.

    Currently supports SEED, XML-SEED and RESP files.
    """
    def __init__(self, cache_db_file, seed_folder, resp_folder,
                 show_progress=True):
        self.index_values = [
            ("channel_id", "TEXT"),
            ("start_date", "INTEGER"),
            ("end_date", "INTEGER"),
            ("latitude", "REAL"),
            ("longitude", "REAL"),
            ("elevation_in_m", "REAL"),
            ("local_depth_in_m", "REAL")]

        self.filetypes = ["seed", "resp"]

        self.seed_folder = seed_folder
        self.resp_folder = resp_folder

        self.__cache_station_coordinates = {}

        super(StationCache, self).__init__(cache_db_file=cache_db_file,
                                           show_progress=show_progress)

    def _find_files_seed(self):
        seed_files = []
        # Get all dataless SEED files.
        for filename in glob.iglob(os.path.join(self.seed_folder,
                                                "dataless.*")):
            seed_files.append(filename)
        return seed_files

    def _find_files_resp(self):
        resp_files = []
        # Get all RESP files
        for filename in glob.iglob(os.path.join(self.resp_folder, "RESP.*")):
            resp_files.append(filename)
        return resp_files

    @staticmethod
    def _extract_index_values_seed(filename):
        """
        Reads SEED files and extracts some keys per channel.
        """
        try:
            p = Parser(filename)
        except:
            msg = "Could not read SEED file '%s'." % filename
            raise ValueError(msg)
        channels = p.getInventory()["channels"]

        channels = [[
            _i["channel_id"], int(_i["start_date"].timestamp),
            int(_i["end_date"].timestamp) if _i["end_date"] else None,
            _i["latitude"], _i["longitude"], _i["elevation_in_m"],
            _i["local_depth_in_m"]] for _i in channels]

        return channels

    @staticmethod
    def _extract_index_values_resp(filename):
        try:
            channels = simple_resp_parser.get_inventory(filename,
                                                        remove_duplicates=True)
        except:
            msg = "Could not read RESP file '%s'." % filename
            raise ValueError(msg)

        channels = [[
            _i["channel_id"], int(_i["start_date"].timestamp),
            int(_i["end_date"].timestamp) if _i["end_date"] else None,
            None, None, None, None] for _i in channels]

        return channels

    def get_coordinates_for_station(self, network, station):
        """
        Returns the coordinates for any station from the cache.

        :param network: Network code.
        :param station: Station code.
        """
        # Caching per instance. Otherwise this is very very slow.
        station_id = network + "." + station

        # Check if something has been written to the instance variable. If
        # yes, assume all available stations have been written to it.
        if self.__cache_station_coordinates:
            try:
                coordinates = self.__cache_station_coordinates[station_id]
            except:
                msg = "No coordinates found for whatever reason."
                raise ValueError(msg)
            else:
                return coordinates

        # Otherwise get all coordinates!
        query = """
        SELECT channel_id, latitude, longitude, elevation_in_m,
            local_depth_in_m
        FROM indices
        WHERE latitude IS NOT NULL
        """
        results = self.db_cursor.execute(query).fetchall()

        stations = {}

        for result in results:
            network_code, station_code = result[0].split(".")[:2]
            station_id = ".".join((network_code, station_code))
            this_station = {
                "latitude": result[1],
                "longitude": result[2],
                "elevation_in_m": result[3],
                "local_depth_in_m": result[4]}

            if station_id in stations:
                if this_station == stations[station_id]:
                    continue
                new_station = stations[station_id]

                # Check if they are equal to a certain tolerance.
                if \
                        abs(this_station["latitude"] -
                            new_station["latitude"]) > TOL_DEGREES \
                        or \
                        abs(this_station["longitude"] -
                            new_station["longitude"]) > TOL_DEGREES \
                        or \
                        abs(this_station["elevation_in_m"] -
                            new_station["elevation_in_m"]) > TOL_METERS \
                        or \
                        abs(this_station["local_depth_in_m"] -
                            new_station["local_depth_in_m"]) > TOL_METERS:
                    msg = ("Several different coordinates set found in "
                           "station cache for %s. The first one found will be "
                           "chosen.") % station_id
                    warnings.warn(msg)
                continue

            stations[station_id] = this_station

        self.__cache_station_coordinates = stations
        # Recurse once; this only accesses the cache.
        return self.get_coordinates_for_station(network, station)

    def get_channels(self):
        """
        Returns a dictionary containing all channels.
        """
        channels = {}
        for channel in self.db_cursor.execute("SELECT * FROM indices")\
                .fetchall():
            channels.setdefault(channel[1], [])
            channels[channel[1]].append({
                "startime_timestamp": channel[2],
                "endtime_timestamp": channel[3],
                "latitude": channel[4],
                "longitude": channel[5],
                "elevation_in_m": channel[6],
                "local_depth_in_m": channel[7]
            })
        return channels

    def get_stations(self):
        """
        Returns a dictionary containing the coordinates of all stations. For
        every station, the first matching channel is chosen and the coordinates
        of the channel are taken.
        """
        stations = {}
        for channel in self.db_cursor.execute("SELECT * FROM indices")\
                .fetchall():
            station_id = ".".join(channel[1].split(".")[:2])
            if station_id in stations:
                continue
            stations[station_id] = {
                "latitude": channel[4],
                "longitude": channel[5],
            }
        return stations

    def get_station_filename(self, channel_id, time):
        """
        Returns the filename for the requested channel and time.

        :param channel_id: The channel id.
        :param time: The time as a timestamp.
        """
        time = int(time.timestamp)
        sql_query = """
        SELECT files.filename FROM indices
        INNER JOIN files
        ON indices.filepath_id=files.id
        WHERE (indices.channel_id = '%s') AND (indices.start_date < %i) AND
            ((indices.end_date IS NULL) OR (indices.end_date > %i))
        LIMIT 1;
        """ % (channel_id, time, time)
        try:
            return self.db_cursor.execute(sql_query).fetchone()[0]
        except sqlite3.Error:
            return None

    def station_info_available(self, channel_id, time):
        """
        Checks if information for the requested channel_id and time is
        available.

        :param channel_id: The channel id.
        :param time: The time as a timestamp.
        """
        try:
            time = int(time.timestamp)
        except AttributeError:
            time = int(time)
        sql_query = """
        SELECT id FROM indices
        WHERE (channel_id = '%s') AND (start_date < %i) AND
            ((end_date IS NULL) OR (end_date > %i))
        LIMIT 1;
        """ % (channel_id, time, time)
        if self.db_cursor.execute(sql_query).fetchone():
            return True
        return False
