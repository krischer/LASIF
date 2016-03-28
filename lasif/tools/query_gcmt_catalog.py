#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import glob
import inspect
import numpy as np
import obspy
from obspy.core.event import Catalog
import os
import random
from scipy.spatial import cKDTree

EARTH_RADIUS = 6371.00

from lasif.utils import get_event_filename


class SphericalNearestNeighbour(object):
    """
    Spherical nearest neighbour queries using scipy's fast
    kd-tree implementation.
    """
    def __init__(self, data):
        cart_data = self.spherical2cartesian(data)
        self.data = data
        self.kd_tree = cKDTree(data=cart_data, leafsize=10)

    def query(self, points, k=10):
        points = self.spherical2cartesian(points)
        d, i = self.kd_tree.query(points, k=k)
        return d, i

    @staticmethod
    def spherical2cartesian(data):
        """
        Converts an array of shape (x, 2) containing latitude/longitude
        pairs into an array of shape (x, 3) containing x/y/z assuming a
        radius of one for points on the surface of a sphere.
        """
        lat = data[:, 0]
        lng = data[:, 1]
        # Convert data from lat/lng to x/y/z, assume radius of 1
        colat = 90 - lat
        cart_data = np.empty((lat.shape[0], 3))

        cart_data[:, 0] = np.sin(np.deg2rad(colat)) * \
            np.cos(np.deg2rad(lng))
        cart_data[:, 1] = np.sin(np.deg2rad(colat)) * \
            np.sin(np.deg2rad(lng))
        cart_data[:, 2] = np.cos(np.deg2rad(colat))

        return cart_data


def _read_GCMT_catalog(min_year=None, max_year=None):
    """
    Helper function reading the GCMT data shipping with LASIF.

    :param min_year: The minimum year to read.
    :type min_year: int, optional
    :param max_year: The maximum year to read.
    :type max_year: int, optional
    """
    # easier tests
    if min_year is None:
        min_year = 0
    else:
        min_year = int(min_year)
    if max_year is None:
        max_year = 3000
    else:
        max_year = int(max_year)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "GCMT_Catalog")
    available_years = [_i for _i in os.listdir(data_dir) if _i.isdigit()]
    available_years.sort()
    print("LASIF currently contains GCMT data from %s to %s/%i." % (
        available_years[0], available_years[-1],
        len(glob.glob(os.path.join(data_dir, available_years[-1], "*.ndk*")))))

    available_years = [_i for _i in os.listdir(data_dir) if _i.isdigit() and
                       (min_year <= int(_i) <= max_year)]
    available_years.sort()

    print("Parsing the GCMT catalog. This might take a while...")
    cat = Catalog()
    for year in available_years:
        print("\tReading year %s ..." % year)
        for filename in glob.glob(os.path.join(data_dir, str(year),
                                               "*.ndk*")):
            cat += obspy.read_events(filename, format="ndk")

    return cat


def add_new_events(comm, count, min_magnitude, max_magnitude, min_year=None,
                   max_year=None, threshold_distance_in_km=50.0):
    min_magnitude = float(min_magnitude)
    max_magnitude = float(max_magnitude)

    # Get the catalog.
    cat = _read_GCMT_catalog(min_year=min_year, max_year=max_year)
    # Filter with the magnitudes
    cat = cat.filter("magnitude >= %.2f" % min_magnitude,
                     "magnitude <= %.2f" % max_magnitude)

    # Filtering catalog to only contain events in the domain.
    print("Filtering to only include events inside domain...")
    # Coordinates and the Catalog will have the same order!
    temp_cat = Catalog()
    coordinates = []
    for event in cat:
        org = event.preferred_origin() or event.origins[0]
        if not comm.query.point_in_domain(org.latitude, org.longitude):
            continue
        temp_cat.events.append(event)
        coordinates.append((org.latitude, org.longitude))
    cat = temp_cat

    chosen_events = []

    print("%i valid events remain. Starting selection process..." % len(cat))

    existing_events = comm.events.get_all_events().values()
    # Get the coordinates of all existing events.
    existing_coordinates = [
        (_i["latitude"], _i["longitude"]) for _i in existing_events]
    existing_origin_times = [_i["origin_time"] for _i in existing_events]

    # Special case handling in case there are no preexisting events.
    if not existing_coordinates:
        idx = random.randint(0, len(cat) - 1)

        chosen_events.append(cat[idx])
        del cat.events[idx]
        existing_coordinates.append(coordinates[idx])
        del coordinates[idx]

        _t = cat[idx].preferred_origin() or cat[idx].origins[0]
        existing_origin_times.append(_t.time)

        count -= 1

    while count:
        if not coordinates:
            print("\tNo events left to select from. Stoping here.")
            break
        # Build kdtree and query for the point furthest away from any other
        # point.
        kdtree = SphericalNearestNeighbour(np.array(existing_coordinates))
        distances = kdtree.query(np.array(coordinates), k=1)[0]
        idx = np.argmax(distances)

        event = cat[idx]
        coods = coordinates[idx]
        del cat.events[idx]
        del coordinates[idx]

        # Actual distance.
        distance = EARTH_RADIUS * distances[idx]

        if distance < threshold_distance_in_km:
            print("\tNo events left with distance to the next closest event "
                  "of more then %.1f km. Stoping here." %
                  threshold_distance_in_km)
            break

        # Make sure it did not happen within one day of an existing event.
        # This should also filter out duplicates.
        _t = event.preferred_origin() or event.origins[0]
        origin_time = _t.time

        if min([abs(origin_time - _i) for _i in existing_origin_times]) < \
                86400:
            print("\tSelected event temporally to close to existing event. "
                  "Will not be chosen. Skipping to next event.")
            continue

        print("\tSelected event with the next closest event being %.1f km "
              "away." % distance)

        chosen_events.append(event)
        existing_coordinates.append(coods)
        count -= 1

    print("Selected %i events." % len(chosen_events))

    folder = comm.project.paths["events"]
    for event in chosen_events:
        filename = os.path.join(folder, get_event_filename(event, "GCMT"))
        Catalog(events=[event]).write(filename, format="quakeml",
                                      validate=True)
        print("Written %s" % (os.path.relpath(filename)))
