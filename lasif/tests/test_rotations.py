#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the rotations.

Attempts to test the rotations as complete as possible as this is a very likely
source of error and tough to get right.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012-2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np

from lasif import rotations


def test_LatLonRadiusToXyz():
    """
    Test the rotations.lat_lon_radius_to_xyz() function.
    """
    # For (0/0)
    np.testing.assert_array_almost_equal(
        rotations.lat_lon_radius_to_xyz(0.0, 0.0, 1.0),
        np.array([1.0, 0.0, 0.0]))
    # At the North Pole
    np.testing.assert_array_almost_equal(
        rotations.lat_lon_radius_to_xyz(90.0, 0.0, 1.0),
        np.array([0.0, 0.0, 1.0]))
    # At the South Pole
    np.testing.assert_array_almost_equal(
        rotations.lat_lon_radius_to_xyz(-90.0, 0.0, 1.0),
        np.array([0.0, 0.0, -1.0]))
    # At the "West Pole"
    np.testing.assert_array_almost_equal(
        rotations.lat_lon_radius_to_xyz(0.0, -90.0, 1.0),
        np.array([0.0, -1.0, 0.0]))
    # At the "East Pole"
    np.testing.assert_array_almost_equal(
        rotations.lat_lon_radius_to_xyz(0.0, 90.0, 1.0),
        np.array([0.0, 1.0, 0.0]))


def test_XyzToLatLonRadius():
    """
    Test the rotations.xyz_to_lat_lon_radius() function.
    """
    # For (0/0)
    lat, lon, radius = rotations.xyz_to_lat_lon_radius(1.0, 0.0, 0.0)
    np.testing.assert_almost_equal(lat, 0.0)
    np.testing.assert_almost_equal(lon, 0.0)
    np.testing.assert_almost_equal(radius, 1.0)
    # At the North Pole
    lat, lon, radius = rotations.xyz_to_lat_lon_radius(0.0, 0.0, 1.0)
    np.testing.assert_almost_equal(lat, 90.0)
    np.testing.assert_almost_equal(lon, 0.0)
    np.testing.assert_almost_equal(radius, 1.0)
    # At the South Pole
    lat, lon, radius = rotations.xyz_to_lat_lon_radius(0.0, 0.0, -1.0)
    np.testing.assert_almost_equal(lat, -90.0)
    np.testing.assert_almost_equal(lon, 0.0)
    np.testing.assert_almost_equal(radius, 1.0)
    # At the "West Pole"
    lat, lon, radius = rotations.xyz_to_lat_lon_radius(0.0, -1.0, 0.0)
    np.testing.assert_almost_equal(lat, 0.0)
    np.testing.assert_almost_equal(lon, -90.0)
    np.testing.assert_almost_equal(radius, 1.0)
    # At the "East Pole"
    lat, lon, radius = rotations.xyz_to_lat_lon_radius(0.0, 1.0, 0.0)
    np.testing.assert_almost_equal(lat, 0.0)
    np.testing.assert_almost_equal(lon, 90.0)
    np.testing.assert_almost_equal(radius, 1.0)
