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
from binascii import crc32
import glob
from obspy.xseed import Parser
import os
import sqlite3

SQL_CREATE_FILES_TABLE = """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        last_modified REAL,
        crc32_hash INTEGER
    );
"""

SQL_CREATE_STATIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS stations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id TEXT,
        start_date INTEGER,
        end_date INTEGER,
        latitude REAL,
        longitude REAL,
        elevation_in_m REAL,
        local_depth_in_m REAL,
        filepath_id INTEGER,
        FOREIGN KEY(filepath_id) REFERENCES files(id) ON DELETE CASCADE
    );
"""


class StationCache(object):
    """
    XXX: Handle RESP files.
    """
    def __init__(self, cache_file, seed_folder, station_xml_folder,
            resp_folder):
        self.cache_file = cache_file

        self.seed_folder = seed_folder
        self.station_xml_folder = station_xml_folder
        self.resp_folder = resp_folder

        self._init_database()
        self._get_file_list()
        self._update_database()

    def __del__(self):
        try:
            self.db_conn.close()
        except:
            pass

    def _get_file_list(self):
        """
        Get a list of all files, order by categories and save it in a
        dictionary.

        The dictionaries will have the filenames as the keys and the
        last-modified time as the values.
        """
        self.files = {}
        self.files["SEED"] = {}
        self.files["RESP"] = {}
        self.files["StationXML"] = {}

        # Get all dataless SEED files.
        for filename in glob.iglob(os.path.join(
                self.seed_folder, "dataless.*")):
            self.files["SEED"][filename] = os.path.getmtime(filename)
        # Get all RESP files
        for filename in glob.iglob(os.path.join(self.resp_folder, "RESP.*")):
            self.files["RESP"][filename] = os.path.getmtime(filename)
        # XXX: Include StationXML as soon as its ready.

    def _init_database(self):
        """
        Inits the database connects, turns on foreign key support and creates
        the tables if they do not already exist.
        """
        # Init database and enable foreign key support.
        self.db_conn = sqlite3.connect(self.cache_file)
        self.db_cursor = self.db_conn.cursor()
        self.db_cursor.execute("PRAGMA foreign_keys = ON;")
        self.db_conn.commit()
        # Make sure that foreign key support has been turned on.
        if self.db_cursor.execute("PRAGMA foreign_keys;").fetchone()[0] != 1:
            try:
                self.db_conn.close()
            except:
                pass
            msg = ("Could not enable foreign key support for SQLite. Please "
                "contact the LASIF developers.")
            raise ValueError(msg)
        # Create the tables.
        self.db_cursor.execute(SQL_CREATE_FILES_TABLE)
        self.db_cursor.execute(SQL_CREATE_STATIONS_TABLE)
        self.db_conn.commit()

    def get_channels(self):
        """
        Returns a dictionary containing all channels.
        """
        channels = {}
        for channel in self.db_cursor.execute("SELECT * FROM stations")\
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
        for channel in self.db_cursor.execute("SELECT * FROM stations")\
                .fetchall():
            station_id = ".".join(channel[1].split(".")[:2])
            if station_id in stations:
                continue
            stations[station_id] = {
                "latitude": channel[4],
                "longitude": channel[5],
            }
        return stations

    def _update_database(self):
        """
        Updates the database.
        """
        # Get all files currently in the database and reshape into a
        # dictionary. The dictionary key is the filename and the value a tuple
        # of (id, last_modified, crc32 hash).
        db_files = self.db_cursor.execute("SELECT * FROM files").fetchall()
        db_files = {_i[1]: (_i[0], _i[2], _i[3]) for _i in db_files}
        # First update the SEED files.
        for seed_file, last_modified in self.files["SEED"].iteritems():
            if seed_file in db_files:
                this_file = db_files[seed_file]
                del db_files[seed_file]
                if last_modified <= this_file[1]:
                    continue
                self._update_seed_file(seed_file, this_file[0])
            else:
                self._update_seed_file(seed_file)

        # Remove all files no longer part of the cache DB.
        for filename in db_files.iterkeys():
            self.db_cursor.execute("DELETE FROM files WHERE filename='%s';" %
                filename)
        self.db_conn.commit()

    def _update_seed_file(self, filename, filepath_id=None):
        """
        Updates or creates a new entry for the given file. If id is given, it
        will be interpreted as an update, otherwise as a fresh record.
        """
        if filepath_id is not None:
            self.db_cursor.execute("DELETE FROM stations WHERE "
                "filepath_id = %i" % filepath_id)
            self.db_conn.commit()
        try:
            p = Parser(filename)
        except:
            try:
                self.db_conn.close()
            except:
                pass
            msg = "Could not read SEED file '%s'." % filename
            raise ValueError(msg)
        channels = p.getInventory()["channels"]
        # Update or insert the new file.
        with open(filename, "rb") as open_file:
            filehash = crc32(open_file.read())
        if filepath_id is not None:
            self.db_cursor.execute("UPDATE files SET last_modified=%f, "
                "crc32_hash=%i WHERE id=%i;" % (os.path.getmtime(filename),
                filehash, filepath_id))
            self.db_conn.commit()
        else:
            self.db_cursor.execute("INSERT into files(filename, last_modified,"
                " crc32_hash) VALUES('%s', %f, %i);" % (
                filename, os.path.getmtime(filename), filehash))
            self.db_conn.commit()
            filepath_id = self.db_cursor.lastrowid
        # Enter all the channels.
        channels = [(_i["channel_id"], int(_i["start_date"].timestamp),
            int(_i["end_date"].timestamp) if _i["end_date"] else None,
            _i["latitude"], _i["longitude"], _i["elevation_in_m"],
            _i["local_depth_in_m"], filepath_id) for _i in channels]
        self.db_conn.executemany("INSERT INTO stations(channel_id, start_date,"
           " end_date, latitude, longitude, elevation_in_m, local_depth_in_m, "
           "filepath_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?)", channels)
        self.db_conn.commit()
