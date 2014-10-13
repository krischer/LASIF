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
    Component dealing with station coordinates from web services or manual
    entry. This is used if station coordinates could not be retrieved by
    other means. The component will attempt to request the station
    coordinates from IRIS and ORFEUS. Each station will only be requested
    once and the result will be stored in the database instance.

    :param db_file: The full path of the inventory SQLITE file.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, db_file, communicator, component_name):
        self._db_file = db_file
        super(InventoryDBComponent, self).__init__(communicator,
                                                   component_name)

    @property
    def _conn(self):
        """
        Lazy init of connection.
        """
        try:
            return self.__connection
        except AttributeError:
            pass
        self.__connection = sqlite3.connect(self._db_file)
        return self.__connection

    @property
    def _cursor(self):
        """
        Lazy init of cursor.
        """
        try:
            return self.__cursor
        except AttributeError:
            pass
        sql = """
        CREATE TABLE IF NOT EXISTS stations(
            station_name TEXT PRIMARY_KEY UNIQUE,
            latitude REAL,
            longitude REAL,
            elevation REAL,
            depth REAL
        );"""
        self.__cursor = self._conn.cursor()
        self.__cursor.execute(sql)
        self._conn.commit()
        return self.__cursor

    def save_station_coordinates(self, station_id, latitude, longitude,
                                 elevation_in_m, local_depth_in_m):
        """
        Saves the coordinates for some station in the database.

        Used internally but can also be used to save coordinates from other
        sources in the project.

        >>> comm = getfixture('inventory_db_comm')
        >>> comm.inventory_db.save_station_coordinates(station_id="XX.YY",
        ... latitude=10.0, longitude=11.0, elevation_in_m=12.0,
        ... local_depth_in_m=13.0)
        >>> comm.inventory_db.get_coordinates("XX.YY") \
        # doctest: +NORMALIZE_WHITESPACE
        {'latitude': 10.0, 'elevation_in_m': 12.0, 'local_depth_in_m': 13.0,
        'longitude': 11.0}

        Inserting once again will update the entry.

        >>> comm.inventory_db.save_station_coordinates(station_id="XX.YY",
        ... latitude=20.0, longitude=21.0, elevation_in_m=22.0,
        ... local_depth_in_m=23.0)
        >>> comm.inventory_db.get_coordinates("XX.YY") \
        # doctest: +NORMALIZE_WHITESPACE
        {'latitude': 20.0, 'elevation_in_m': 22.0, 'local_depth_in_m': 23.0,
         'longitude': 21.0}
        """
        # Either all are None or only the local_depth.
        coods = [latitude, longitude, elevation_in_m, local_depth_in_m]
        if None in coods:
            count = coods.count(None)
            if count != 4 and local_depth_in_m is not None:
                msg = "Only local depth is allowed to be None."
                raise ValueError(msg)

        latitude = str(latitude) if latitude is not None else None
        longitude = str(longitude) if longitude is not None else None
        elevation_in_m = str(elevation_in_m) \
            if elevation_in_m is not None else None
        local_depth_in_m = str(local_depth_in_m) \
            if local_depth_in_m is not None else None

        sql = """
        INSERT OR REPLACE INTO stations
            (station_name, latitude, longitude, elevation, depth)
        VALUES (?, ?, ?, ?, ?);
        """
        self._cursor.execute(sql, (station_id, latitude, longitude,
                                   elevation_in_m, local_depth_in_m))
        self._conn.commit()

    def remove_coordinate_less_stations(self):
        """
        Simple command removing all stations that have no associated
        coordinates. This means it will attempt to download them again on
        the next run.

        >>> comm = getfixture('inventory_db_comm')
        >>> comm.inventory_db.get_all_coordinates() \
        # doctest: +NORMALIZE_WHITESPACE
        {u'AA.BB': {'latitude': 1.0, 'elevation_in_m': 3.0,
                    'local_depth_in_m': 4.0, 'longitude': 2.0},
         u'CC.DD': {'latitude': 2.0, 'elevation_in_m': 2.0,
                    'local_depth_in_m': 2.0, 'longitude': 2.0},
         u'EE.FF': {'latitude': None, 'elevation_in_m': None,
                    'local_depth_in_m': None, 'longitude': None}}
        >>> comm.inventory_db.remove_coordinate_less_stations()
        >>> comm.inventory_db.get_all_coordinates() \
        # doctest: +NORMALIZE_WHITESPACE
        {u'AA.BB': {'latitude': 1.0, 'elevation_in_m': 3.0,
                    'local_depth_in_m': 4.0, 'longitude': 2.0},
         u'CC.DD': {'latitude': 2.0, 'elevation_in_m': 2.0,
                    'local_depth_in_m': 2.0, 'longitude': 2.0}}
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
        inventory database. All entries will be ``None`` if a webservice
        request has been attempted but returned nothing. This is to prevent
        successive requests.

        >>> comm = getfixture('inventory_db_comm')
        >>> comm.inventory_db.get_all_coordinates() \
        # doctest: +NORMALIZE_WHITESPACE
        {u'AA.BB': {'latitude': 1.0, 'elevation_in_m': 3.0,
                    'local_depth_in_m': 4.0, 'longitude': 2.0},
         u'CC.DD': {'latitude': 2.0, 'elevation_in_m': 2.0,
                    'local_depth_in_m': 2.0, 'longitude': 2.0},
         u'EE.FF': {'latitude': None, 'elevation_in_m': None,
                    'local_depth_in_m': None, 'longitude': None}}
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
        Returns a dictionary containing ``latitude``, ``longitude``,
        ``elevation_in_m``, and ``local_depth_in_m`` keys. If the station is
        not yet in the database, it will attempt to download the coordinates
        from the web services.

        >>> comm = getfixture('inventory_db_comm')
        >>> comm.inventory_db.get_coordinates("AA.BB") \
        # doctest: +NORMALIZE_WHITESPACE
        {'latitude': 1.0, 'elevation_in_m': 3.0, 'local_depth_in_m': 4.0,
         'longitude': 2.0}

        A station whose coordinates have already been requested but whose
        request failed will have ``None`` for all coordinate values.

        >>> comm.inventory_db.get_coordinates("EE.FF") \
        # doctest: +NORMALIZE_WHITESPACE
        {'latitude': None, 'elevation_in_m': None, 'local_depth_in_m': None,
         'longitude': None}
        """
        sql = """
        SELECT latitude, longitude, elevation, depth
        FROM stations
        WHERE station_name = ?
        LIMIT 1;
        """
        coordinates = self._cursor.execute(sql, (station_id,)).fetchone()

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
        try:
            lat = float(re.findall("<Latitude>(.*)</Latitude>",
                                   request_text)[0])
        except ValueError:
            lat = None

        try:
            lng = float(re.findall("<Longitude>(.*)</Longitude>",
                                   request_text)[0])
        except ValueError:
            lng = None

        try:
            ele = float(re.findall("<Elevation>(.*)</Elevation>",
                                   request_text)[0])
        except ValueError:
            ele = None

        # The local is not set at the station level.
        self.save_station_coordinates(station_id, lat, lng, ele, None)
        print "Success."
        return {"latitude": lat, "longitude": lng, "elevation_in_m": ele,
                "local_depth_in_m": None}
