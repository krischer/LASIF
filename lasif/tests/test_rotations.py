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


def test_RotateVector():
    """
    Test the basic vector rotation around an arbitrary axis.
    """
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([0, 0, 1], [0, 1, 0], 90),
        np.array([1.0, 0.0, 0.0]), decimal=10)
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([0, 0, 1], [0, 1, 0], 180),
        np.array([0.0, 0.0, -1.0]), decimal=10)
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([0, 0, 1], [0, 1, 0], 270),
        np.array([-1.0, 0.0, 0.0]), decimal=10)
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([0, 0, 1], [0, 1, 0], 360),
        np.array([0.0, 0.0, 1.0]), decimal=10)
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([0, 0, 1], [0, 1, 0], 45),
        np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)]), decimal=10)
    # Use a different vector and rotation angle.
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([1, 1, 1], [1, 0, 0], 90),
        np.array([1.0, -1.0, 1.0]), decimal=10)
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([1, 1, 1], [1, 0, 0], 180),
        np.array([1.0, -1.0, -1.0]), decimal=10)
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([1, 1, 1], [1, 0, 0], 270),
        np.array([1.0, 1.0, -1.0]), decimal=10)
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([1, 1, 1], [1, 0, 0], 360),
        np.array([1.0, 1.0, 1.0]), decimal=10)
    # Some more arbitrary rotations.
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([234.3, -5645.4, 345.45], [1, 0, 0], 360),
        np.array([234.3, -5645.4, 345.45]), decimal=10)
    np.testing.assert_array_almost_equal(
        rotations.rotate_vector([234.3, -5645.4, 345.45], [1, 0, 0], 180),
        np.array([234.3, 5645.4, -345.45]), decimal=10)


def test_GetSphericalUnitVectors():
    """
    Tests the rotations.get_spherical_unit_vectors() function.
    """
    # At 0, 0
    e_theta, e_phi, e_r = rotations.get_spherical_unit_vectors(0.0, 0.0)
    np.testing.assert_array_almost_equal(
        e_theta, np.array([0, 0, -1]))
    np.testing.assert_array_almost_equal(
        e_phi, np.array([0, 1, 0]))
    np.testing.assert_array_almost_equal(
        e_r, np.array([1, 0, 0]))
    # At the north pole
    e_theta, e_phi, e_r = rotations.get_spherical_unit_vectors(90.0, 0.0)
    np.testing.assert_array_almost_equal(
        e_theta, np.array([1, 0, 0]))
    np.testing.assert_array_almost_equal(
        e_phi, np.array([0, 1, 0]))
    np.testing.assert_array_almost_equal(
        e_r, np.array([0, 0, 1]))
    # At the south pole
    e_theta, e_phi, e_r = rotations.get_spherical_unit_vectors(-90.0, 0.0)
    np.testing.assert_array_almost_equal(
        e_theta, np.array([-1, 0, 0]))
    np.testing.assert_array_almost_equal(
        e_phi, np.array([0, 1, 0]))
    np.testing.assert_array_almost_equal(
        e_r, np.array([0, 0, -1]))
    # At the "east pole"
    e_theta, e_phi, e_r = rotations.get_spherical_unit_vectors(0.0, 90.0)
    np.testing.assert_array_almost_equal(
        e_theta, np.array([0, 0, -1]))
    np.testing.assert_array_almost_equal(
        e_phi, np.array([-1, 0, 0]))
    np.testing.assert_array_almost_equal(
        e_r, np.array([0, 1, 0]))
    # At the "west pole"
    e_theta, e_phi, e_r = rotations.get_spherical_unit_vectors(0.0, -90.0)
    np.testing.assert_array_almost_equal(
        e_theta, np.array([0, 0, -1]))
    np.testing.assert_array_almost_equal(
        e_phi, np.array([1, 0, 0]))
    np.testing.assert_array_almost_equal(
        e_r, np.array([0, -1, 0]))


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


def test_RotateLatLon():
    """
    Test the lat/lon rotation on a sphere.
    """
    # Rotate north pole to equator.
    lat_new, lon_new = rotations.rotate_lat_lon(90.0, 0.0, [0, 1, 0], 90)
    np.testing.assert_almost_equal(lat_new, 0.0)
    np.testing.assert_almost_equal(lon_new, 0.0)
    # Rotate north pole to the south pole.
    lat_new, lon_new = rotations.rotate_lat_lon(90.0, 0.0, [0, 1, 0], 180)
    np.testing.assert_almost_equal(lat_new, -90.0)
    np.testing.assert_almost_equal(lon_new, 0.0)
    # Rotate north pole to equator, the other way round.
    lat_new, lon_new = rotations.rotate_lat_lon(90.0, 0.0, [0, 1, 0], -90)
    np.testing.assert_almost_equal(lat_new, 0.0)
    np.testing.assert_almost_equal(lon_new, 180.0)
    # Rotate (0/0) to the east
    lat_new, lon_new = rotations.rotate_lat_lon(0.0, 0.0, [0, 0, 1], 90)
    np.testing.assert_almost_equal(lat_new, 0.0)
    np.testing.assert_almost_equal(lon_new, 90.0)
    # Rotate (0/0) to the west
    lat_new, lon_new = rotations.rotate_lat_lon(0.0, 0.0, [0, 0, 1], -90)
    np.testing.assert_almost_equal(lat_new, 0.0)
    np.testing.assert_almost_equal(lon_new, -90.0)
    # Rotate the west to the South Pole. The longitude can not be tested
    # reliably because is varies infinitly fast directly at a pole.
    lat_new, lon_new = rotations.rotate_lat_lon(0.0, -90.0, [1, 0, 0], 90)
    np.testing.assert_almost_equal(lat_new, -90.0)


def test_RotateData():
    """
    Test the rotations.rotate_data() function.
    """
    north_data = np.linspace(0, 10, 20)
    east_data = np.linspace(33, 44, 20)
    vertical_data = np.linspace(-12, -34, 20)
    # A rotation around the rotation axis of the earth with a source at the
    # equator should not change anything.
    new_north_data, new_east_data, new_vertical_data = \
        rotations.rotate_data(north_data, east_data, vertical_data, 0.0,
                              123.45, [0, 0, 1], 77.7)
    np.testing.assert_array_almost_equal(north_data, new_north_data, 5)
    np.testing.assert_array_almost_equal(east_data, new_east_data, 5)
    np.testing.assert_array_almost_equal(vertical_data, new_vertical_data,
                                         5)
    # A rotation around the rotation axis of the earth should not change
    # the vertical component.
    new_north_data, new_east_data, new_vertical_data = \
        rotations.rotate_data(north_data, east_data, vertical_data, -55.66,
                              123.45, [0, 0, 1], 77.7)
    np.testing.assert_array_almost_equal(vertical_data, new_vertical_data,
                                         5)
    # The same is true for any other rotation with an axis through the
    # center of the earth.
    new_north_data, new_east_data, new_vertical_data = \
        rotations.rotate_data(north_data, east_data, vertical_data, -55.66,
                              123.45, [123, 345.0, 0.234], 77.7)
    np.testing.assert_array_almost_equal(vertical_data, new_vertical_data,
                                         5)
    # Any data along the Greenwich meridian and the opposite one should not
    # change with a rotation around the "East Pole" or the "West Pole".
    new_north_data, new_east_data, new_vertical_data = \
        rotations.rotate_data(north_data, east_data, vertical_data, 0.0,
                              0.0, [0, 1, 0], 55.0)
    np.testing.assert_array_almost_equal(north_data, new_north_data, 5)
    np.testing.assert_array_almost_equal(east_data, new_east_data, 5)
    np.testing.assert_array_almost_equal(vertical_data, new_vertical_data,
                                         5)
    new_north_data, new_east_data, new_vertical_data = \
        rotations.rotate_data(north_data, east_data, vertical_data, 0.0,
                              0.0, [0, -1, 0], 55.0)
    np.testing.assert_array_almost_equal(north_data, new_north_data, 5)
    np.testing.assert_array_almost_equal(east_data, new_east_data, 5)
    np.testing.assert_array_almost_equal(vertical_data, new_vertical_data,
                                         5)
    # A rotation of one hundred degree around the x-axis inverts (in this
    # case) north and east components.
    new_north_data, new_east_data, new_vertical_data = \
        rotations.rotate_data(north_data, east_data, vertical_data, 0.0,
                              90.0, [1, 0, 0], 180.0)
    np.testing.assert_array_almost_equal(north_data, -new_north_data, 5)
    np.testing.assert_array_almost_equal(east_data, -new_east_data, 5)
    np.testing.assert_array_almost_equal(vertical_data, new_vertical_data,
                                         5)


def test_RotateMomentTensor():
    """
    Tests the moment tensor rotations.
    """
    # A full rotation should not change anything.
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotations.rotate_moment_tensor(
        1, 2, 3, 4, 5, 6, 7, 8, [9, 10, 11], 360)
    np.testing.assert_almost_equal(Mrr, 1.0, 6)
    np.testing.assert_almost_equal(Mtt, 2.0, 6)
    np.testing.assert_almost_equal(Mpp, 3.0, 6)
    np.testing.assert_almost_equal(Mrt, 4.0, 6)
    np.testing.assert_almost_equal(Mrp, 5.0, 6)
    np.testing.assert_almost_equal(Mtp, 6.0, 6)

    # The following ones are tested against a well proven Matlab script by
    # Andreas Fichtner.
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotations.rotate_moment_tensor(
        -0.704, 0.071, 0.632, 0.226, -0.611, 3.290,
        rotations.colat2lat(26.08), -21.17,
        [0, 1, 0], 57.5)
    Mrr_new, Mtt_new, Mpp_new, Mrt_new, Mrp_new, Mtp_new = \
        [-0.70400000, 2.04919171, -1.34619171, 0.02718681, -0.65089007,
         2.83207047]
    np.testing.assert_almost_equal(Mrr, Mrr_new, 6)
    np.testing.assert_almost_equal(Mtt, Mtt_new, 6)
    np.testing.assert_almost_equal(Mpp, Mpp_new, 6)
    np.testing.assert_almost_equal(Mrt, Mrt_new, 6)
    np.testing.assert_almost_equal(Mrp, Mrp_new, 6)
    np.testing.assert_almost_equal(Mtp, Mtp_new, 6)
    # Another example.
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotations.rotate_moment_tensor(
        -0.818, -1.300, 2.120, 1.720, 2.290, -0.081,
        rotations.colat2lat(53.51), -9.87,
        [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0], -31.34)
    Mrr_new, Mtt_new, Mpp_new, Mrt_new, Mrp_new, Mtp_new = \
        [-0.81800000, -0.69772178, 1.51772178, 2.55423451, 1.29552541,
         1.30522545]
    np.testing.assert_almost_equal(Mrr, Mrr_new, 6)
    np.testing.assert_almost_equal(Mtt, Mtt_new, 6)
    np.testing.assert_almost_equal(Mpp, Mpp_new, 6)
    np.testing.assert_almost_equal(Mrt, Mrt_new, 6)
    np.testing.assert_almost_equal(Mrp, Mrp_new, 6)
    np.testing.assert_almost_equal(Mtp, Mtp_new, 6)
    # The same as before, but with a non-normalized axis. Should work just
    # as well.
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotations.rotate_moment_tensor(
        -0.818, -1.300, 2.120, 1.720, 2.290, -0.081,
        rotations.colat2lat(53.51), -9.87,
        [11.12, -11.12, 0], -31.34)
    Mrr_new, Mtt_new, Mpp_new, Mrt_new, Mrp_new, Mtp_new = \
        [-0.81800000, -0.69772178, 1.51772178, 2.55423451, 1.29552541,
         1.30522545]
    np.testing.assert_almost_equal(Mrr, Mrr_new, 6)
    np.testing.assert_almost_equal(Mtt, Mtt_new, 6)
    np.testing.assert_almost_equal(Mpp, Mpp_new, 6)
    np.testing.assert_almost_equal(Mrt, Mrt_new, 6)
    np.testing.assert_almost_equal(Mrp, Mrp_new, 6)
    np.testing.assert_almost_equal(Mtp, Mtp_new, 6)
    # One more.
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotations.rotate_moment_tensor(
        0.952, -1.030, 0.076, 0.226, -0.040, -0.165,
        rotations.colat2lat(63.34), 55.80,
        [np.sqrt(3) / 3, -np.sqrt(3) / 3, -np.sqrt(3) / 3], 123.45)
    Mrr_new, Mtt_new, Mpp_new, Mrt_new, Mrp_new, Mtp_new = \
        [0.95200000, -0.41458722, -0.53941278, -0.09170855, 0.21039378,
         0.57370606]
    np.testing.assert_almost_equal(Mrr, Mrr_new, 6)
    np.testing.assert_almost_equal(Mtt, Mtt_new, 6)
    np.testing.assert_almost_equal(Mpp, Mpp_new, 6)
    np.testing.assert_almost_equal(Mrt, Mrt_new, 6)
    np.testing.assert_almost_equal(Mrp, Mrp_new, 6)
    np.testing.assert_almost_equal(Mtp, Mtp_new, 6)
