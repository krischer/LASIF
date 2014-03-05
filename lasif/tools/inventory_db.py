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
import cPickle
import obspy
import obspy.arclink
import os
import re
import sqlite3
import time
import urllib2


CREATE_DB_SQL = """
CREATE TABLE IF NOT EXISTS stations(
    station_name TEXT,
    latitude REAL,
    longitude REAL,
    elevation REAL,
    depth REAL
);"""


URL = ("http://service.iris.edu/fdsnws/station/1/query?"
       "network={network}&sta={station}&level=station&nodata=404")


class InventoryDB(object):
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.cursor.execute(CREATE_DB_SQL)
        self.conn.commit()

    def __del__(self):
        try:
            self.conn.close()
        except:
            pass

    def put_station_coordinates(self, station_id, latitude, longitude,
                                elevation_in_m, depth_in_m):
        latitude = str(latitude) if latitude is not None else "NULL"
        longitude = str(longitude) if longitude is not None else "NULL"
        elevation_in_m = str(elevation_in_m) \
            if elevation_in_m is not None else "NULL"
        depth_in_m = str(depth_in_m) if depth_in_m is not None else "NULL"

        SQL = """
        REPLACE INTO stations
            (station_name, latitude, longitude, elevation, depth)
        VALUES ('%s', %s, %s, %s, %s);
        """ % (station_id, latitude, longitude, elevation_in_m, depth_in_m)
        self.cursor.execute(SQL)
        self.conn.commit()


def reset_coordinate_less_stations(db_file):
    """
    Simple command removing all stations that have no associated coordinates.

    :param db_file: The SQLite database filepath.
    """
    inv_db = InventoryDB(db_file)

    SQL = """
    DELETE FROM stations
    WHERE latitude is null;
    """
    inv_db.cursor.execute(SQL)
    inv_db.conn.commit()


def get_station_coordinates(db_file, station_id, cache_folder, arclink_user):
    """
    Returns either a dictionary containing "latitude", "longitude",
    "elevation_in_m", "local_depth_in_m" keys or None if nothing was found.
    """
    inv_db = InventoryDB(db_file)

    SQL = """
    SELECT latitude, longitude, elevation, depth
    FROM stations
    WHERE station_name = '%s'
    LIMIT 1;
    """ % station_id
    coordinates = inv_db.cursor.execute(SQL).fetchone()

    if coordinates and coordinates[0] is None:
        return None

    elif coordinates:
        return {"latitude": coordinates[0], "longitude": coordinates[1],
                "elevation_in_m": coordinates[2],
                "local_depth_in_m": coordinates[3]}

    msg = ("Attempting to download coordinates for %s. This will only "
           "happen once ... ") % station_id
    print msg,
    # Otherwise try to download the necessary information.
    network, station = station_id.split(".")
    req = None
    for _i in xrange(10):
        try:
            req = urllib2.urlopen(URL.format(network=network, station=station))
            break
        except:
            time.sleep(0.1)
    if req is None or str(req.code)[0] != "2":
        # Now also attempt to download via ArcLink.
        pickled_inventory = os.path.join(cache_folder, "arclink_inv.pickle")
        # Download all 30 days...
        if not os.path.exists(pickled_inventory) or (
                (time.time() - os.path.getmtime(pickled_inventory))
                > 30 * 86500):
            c = obspy.arclink.Client(user=arclink_user)
            try:
                inv = c.getNetworks(obspy.UTCDateTime(1970),
                                    obspy.UTCDateTime())
                inv = {key: value for (key, value) in inv.iteritems()
                       if len(key.split('.')) == 2}
                with open(pickled_inventory, "wb") as fh:
                    cPickle.dump(inv, fh)
            except:
                msg = ("Failed to download ArcLink Inventory. If the problem "
                       "persits, contact the developers.")
                raise Exception(msg)
                inv = None
        else:
            with open(pickled_inventory, "rb") as fh:
                inv = cPickle.load(fh)
        if inv is None or station_id not in inv:
            print "Failure."
            inv_db.put_station_coordinates(station_id, None, None, None, None)
            return None
        stat = inv[station_id]
        lat = stat.latitude
        lng = stat.longitude
        ele = stat.elevation
        depth = stat.depth
        inv_db.put_station_coordinates(station_id, lat, lng, ele, depth)
        print "Success."
        return {"latitude": lat, "longitude": lng, "elevation_in_m": ele,
                "local_depth_in_m": depth}

    # Now simply find the coordinates.
    request_text = req.read()
    lat = float(re.findall("<Latitude>(.*)</Latitude>", request_text)[0])
    lng = float(re.findall("<Longitude>(.*)</Longitude>", request_text)[0])
    ele = float(re.findall("<Elevation>(.*)</Elevation>", request_text)[0])

    inv_db.put_station_coordinates(station_id, lat, lng, ele, None)
    print "Success."
    return {"latitude": lat, "longitude": lng, "elevation_in_m": ele,
            "local_depth_in_m": 0.0}
