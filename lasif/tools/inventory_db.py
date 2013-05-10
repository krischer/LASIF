#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple query functions for the inventory database.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import inspect
import os
import sqlite3

# Most generic way to get the actual data directory.
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data", "inventory.sqlite")


def get_station_coordinates(station_id):
    """
    Returns either a dictionary containing "latitude", "longitude",
    "elevation_in_m", "local_depth_in_m" keys or None if nothing was found.
    """
    SQL = """
    SELECT latitude, longitude, elevation, depth
    FROM stations
    WHERE station_name = '%s'
    LIMIT 1;
    """ % station_id
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    coordinates = cursor.execute(SQL).fetchone()
    if not coordinates:
        return None
    return {"latitude": coordinates[0], "longitude": coordinates[1],
        "elevation_in_m": coordinates[2], "local_depth_in_m": coordinates[3]}
