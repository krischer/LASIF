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

SQL_CREATE_TABLES = """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        last_modified REAL,
        crc32_hash INTEGER
    );

    CREATE TABLE IF NOT EXISTS stations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id TEXT,
        start_date INTEGER,
        end_date INTEGER,
        FOREIGN KEY(filepath_id) REFERENCES files(id) NOT NULL ON DELETE
            CASCADE
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
        self.db_cursor.commit()
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
        self.db_cursor.execute(SQL_CREATE_TABLES)
        self.db_cursor.commit()

    def update_database(self):
        """
        Updates the database.
        """
        # Get all files currently in the database and reshape into a
        # dictionary. The dictionary key is the filename and the value a tuple
        # of (id, last_modified, crc32 hash).
        db_files = self.db_cursor.commit("SELECT * FROM files").fetchall()
        db_files = {_i[1]: (_i[0], _i[2], _i[3]) for _i in db_files}
        # First update the SEED files.
        for seed_file, last_modified in self.files["SEED"].iteritems():
            if seed_file in db_files:
                if last_modified >= db_files[seed_file][1]:
                    continue
                self._update_seed_file(seed_file, db_files[seed_file][0])
            else:
                self._update_seed_file(seed_file)

    def _update_seed_file(self, filename, filepath_id=None):
        """
        Updates or creates a new entry for the given file. If id is given, it
        will be interpreted as an update, otherwise as a fresh record.
        """
        if filepath_id is not None:
            self.db_cursor("DELETE FROM stations WHERE filepath_id = %i" %
                filepath_id)
            self.db_cursor.commit()
        try:
            p = Parser(filename)
        except:
            try:
                self.db_conn.close()
            except:
                pass
            msg = "Could not read SEED file '%s'." % filename
            raise ValueError(msg)
        channels = p.getInvententory()["channels"]
        # Update or insert the new file.
        if filepath_id is None:
            self.db_cursor.execute("UPDATE files SET last_modified=%f "
                "crc32_hash=%i WHERE id=%i;" % (os.path.getmtime(filename),
                crc32(filename)))
        else:
            self.db_cursor.execute("INSERT into files(filename, last_modified,"
                " crc32_hash) VALUES('%s', %f, %i);" % (
                filename, os.path.getmtime(filename), crc32(filename)))
        self.db_cursor.commit()
        last_id = self.db_cursor.lastrowid
        # Enter all the channels.
        channels = [(_i["channel_id"], int(_i["start_date"].timestamp),
            int(_i["end_date"].timestamp, last_id) if _i["end_date"] else None)
            for _i in channels]
        self.db_conn.executemany("INSERT INTO stations VALUES(?, ?, ?, ?)",
            channels)
        self.db_conn.commit()
