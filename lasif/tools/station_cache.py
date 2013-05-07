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

from lasif.tools import simple_resp_parser
from lasif.tools.file_info_cache import FileInfoCache


class StationCache(FileInfoCache):
    """
    Cache for Station files.

    Currently supports SEED, XML-SEED and RESP files.
    """
    def __init__(self, cache_db_file, seed_folder, resp_folder):
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

        super(StationCache, self).__init__(cache_db_file=cache_db_file)

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

    def _extract_index_values_seed(self, filename):
        """
        Reads SEED files and extracts some keys per channel.
        """
        try:
            p = Parser(filename)
        except:
            msg = "Could not read SEED file '%s'." % filename
            raise ValueError(msg)
        channels = p.getInventory()["channels"]

        channels = [[_i["channel_id"], int(_i["start_date"].timestamp),
            int(_i["end_date"].timestamp) if _i["end_date"] else None,
            _i["latitude"], _i["longitude"], _i["elevation_in_m"],
            _i["local_depth_in_m"]] for _i in channels]

        return channels

    def _extract_index_values_resp(self, filename):
        try:
            channels = simple_resp_parser.get_inventory(filename,
                remove_duplicates=False)
        except:
            msg = "Could not read RESP file '%s'." % filename
            raise ValueError(msg)

        channels = [[_i["channel_id"], int(_i["start_date"].timestamp),
            int(_i["end_date"].timestamp) if _i["end_date"] else None,
            None, None, None, None] for _i in channels]

        return channels

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
