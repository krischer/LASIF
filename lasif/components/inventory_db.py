#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re
import sqlite3
import urllib2

from .component import Component

URL = ("http://{service}/fdsnws/station/1/query?"
       "network={network}&sta={station}&level=station&nodata=404")


class InventoryDBComponent(Component):
    """
    Component wrapping the inventory DB.
    """
    def __init__(self, db_file, communicator, component_name):
        self._db_file = db_file

        sql = """
        CREATE TABLE IF NOT EXISTS stations(
            station_name TEXT,
            latitude REAL,
            longitude REAL,
            elevation REAL,
            depth REAL
        );"""

        self._conn = sqlite3.connect(db_file)
        self._cursor = self._conn.cursor()
        self._cursor.execute(sql)
        self._conn.commit()

        super(InventoryDBComponent, self).__init__(communicator,
                                                   component_name)

    def save_station_coordinates(self, station_id, latitude, longitude,
                                 elevation_in_m, local_depth_in_m):
        """
        Saves the coordinates for some station in the database.
        """
        # Either all are None or only the local_depth.
        coods = [latitude, longitude, elevation_in_m, local_depth_in_m]
        if None in coods:
            count = coods.count(None)
            if count != 4 and local_depth_in_m is not None:
                msg = "Only local depth is allowed to be None."
                raise ValueError(msg)

        latitude = str(latitude) if latitude is not None else "NULL"
        longitude = str(longitude) if longitude is not None else "NULL"
        elevation_in_m = str(elevation_in_m) \
            if elevation_in_m is not None else "NULL"
        local_depth_in_m = str(local_depth_in_m) \
            if local_depth_in_m is not None else "NULL"

        sql = """
        REPLACE INTO stations
            (station_name, latitude, longitude, elevation, depth)
        VALUES ('%s', %s, %s, %s, %s);
        """ % (station_id, latitude, longitude, elevation_in_m,
               local_depth_in_m)
        self._cursor.execute(sql)
        self._conn.commit()

    def remove_coordinate_less_stations(self):
        """
        Simple command removing all stations that have no associated
        coordinates. This means it will attempt to download them again on
        the next run.
        """
        sql = """
        DELETE FROM stations
        WHERE latitude is null;
        """
        self._cursor.execute(sql)
        self._conn.commit()

    def get_all_coordinates(self):
        """
        Returns a dictionary with all stations coordinates defined in the
        inventory database.

        >>> comm.inventory_db.get_all_coordinates()  # doctest: +SKIP
        {'BW.ROTZ': {'latitude': ..., 'longitude': ..,
                     'elevation_in_m': ..., 'local_depth_in_m': ...},
          'AU.INV': {...},
          # All entries will be None if a webservice request has been
          # attempted but returned nothing. This is to prevent successive
          # requests.
          'AU.INV2': {'latitude': None, 'longitude': None,
                      'elevation_in_m': None, 'local_depth_in_m': None},
          ...
        }
        """
        sql = """
        SELECT station_name, latitude, longitude, elevation, depth
        FROM stations
        """
        results = self._cursor.execute(sql).fetchall()

        return {_i[0]: {
            "latitude": _i[1],
            "longitude": _i[2],
            "elevation_in_m": _i[3],
            "local_depth_in_m": _i[4]}
            for _i in results}

    def get_coordinates(self, station_id):
        """
        Returns either a dictionary containing "latitude", "longitude",
        "elevation_in_m", "local_depth_in_m" keys. If the station is in the
        DB but has no coordinates, all values will be set to None.
        """
        sql = """
        SELECT latitude, longitude, elevation, depth
        FROM stations
        WHERE station_name = '%s'
        LIMIT 1;
        """ % station_id
        coordinates = self._cursor.execute(sql).fetchone()

        # If there is a result and it is None, then the coordinate has been
        # requested before but has not been found. Thus return 0.
        if coordinates and coordinates[0] is None:
            return {"latitude": None, "longitude": None,
                    "elevation_in_m": None,
                    "local_depth_in_m": None}
        elif coordinates:
            return {"latitude": coordinates[0], "longitude": coordinates[1],
                    "elevation_in_m": coordinates[2],
                    "local_depth_in_m": coordinates[3]}

        # Otherwise try to download the necessary information.
        msg = ("Attempting to download coordinates for %s. This will only "
               "happen once ... ") % station_id
        print msg,

        req = None
        network, station = station_id.split(".")
        # Try IRIS first.
        try:
            req = urllib2.urlopen(URL.format(
                service="service.iris.edu", network=network, station=station))
        except:
            pass
        # Then ORFEUS.
        if req is None or str(req.code)[0] != "2":
            try:
                req = urllib2.urlopen(URL.format(
                    service="www.orfeus-eu.org", network=network,
                    station=station))
            except:
                pass
        # Otherwise write None's to the database.
        if req is None or str(req.code)[0] != "2":
            self.save_station_coordinates(station_id, None, None, None, None)
            print "Failure."
            return {"latitude": None, "longitude": None,
                    "elevation_in_m": None, "local_depth_in_m": None}

        # Now simply find the coordinates.
        request_text = req.read()
        lat = float(re.findall("<Latitude>(.*)</Latitude>", request_text)[0])
        lng = float(re.findall("<Longitude>(.*)</Longitude>", request_text)[0])
        ele = float(re.findall("<Elevation>(.*)</Elevation>", request_text)[0])

        # The local is not set at the station level.
        self.save_station_coordinates(station_id, lat, lng, ele, None)
        print "Success."
        return {"latitude": lat, "longitude": lng, "elevation_in_m": ele,
                "local_depth_in_m": None}