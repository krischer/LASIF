#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A collection of functions to rotate vectors, seismograms and moment tensors on
a spherical body, e.g. the Earth.

.. note:: **On the used coordinate system**

    Latitude and longitude are natural geographical coordinates used on Earth.

    The coordinate system is right handed with the origin at the center of the
    Earth. The z-axis points directly at the North Pole and the x-axis points
    at (latitude 0.0/longitude 0.0), e.g. the Greenwich meridian at the
    equator. The y-axis therefore points at (latitude 0.0/longitue 90.0), e.g.
    somewhere close to Sumatra.

    𝜃 (theta) is the colatitude, e.g. 90.0 - latitude and is the angle from
    the z-axis.  𝜑 (phi) is the longitude and the angle from the x-axis
    towards the y-axis, a.k.a the azimuth angle. These are also the generally
    used spherical coordinates.

    All rotation axes have to be given as [x, y, z] in the just described
    coordinate system and all rotation angles have to given as degree. A
    positive rotation will rotate clockwise when looking in the direction of
    the rotation axis.

    For convenience reasons, most function in this module work with coordinates
    given in latitude and longitude.

:copyright:
Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012-2013

:license:
GNU Lesser General Public License, Version 3
(http://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np


def _get_vector(*args):
    """
    Helper function to make sure vectors always have the same format and dtype.

    Creates a three component column vector from either a list or three single
    numbers. If it already is a correct vector, do nothing.

    >>> vec = _get_vector(1, 2, 3)
    >>> vec
    array([ 1.,  2.,  3.])
    >>> print vec.dtype
    float64

    >>> vec = _get_vector([1, 2, 3])
    >>> vec
    array([ 1.,  2.,  3.])
    >>> print vec.dtype
    float64

    >>> vec = _get_vector(np.array([1, 2, 3], dtype="int32"))
    >>> vec
    array([ 1.,  2.,  3.])
    >>> print vec.dtype
    float64
    """
    if len(args) == 1 and isinstance(args[0], np.ndarray):
        return np.require(args[0], dtype="float64")
    elif len(args) == 1 and len(args[0]) == 3:
        return np.array(args[0], dtype="float64")
    elif len(args) == 3:
        return np.array(args, dtype="float64")
    else:
        raise NotImplementedError


def lat2colat(lat):
    """
    Helper function to convert latitude to colatitude. This, surprisingly, is
    quite an error source.

    >>> lat2colat(-90)
    180.0
    >>> lat2colat(-45)
    135.0
    >>> lat2colat(0)
    90.0
    >>> lat2colat(45)
    45.0
    >>> lat2colat(90)
    0.0
    """
    return 90.0 - lat


def colat2lat(colat):
    """
    Helper function to convert colatitude to latitude. This, surprisingly, is
    quite an error source.

    >>> colat2lat(180)
    -90.0
    >>> colat2lat(135)
    -45.0
    >>> abs(colat2lat(90))
    0.0
    >>> colat2lat(45)
    45.0
    >>> colat2lat(0.0)
    90.0
    """
    return -1.0 * (colat - 90.0)


def rotate_vector(vector, rotation_axis, angle):
    """
    Takes a vector and rotates it around a rotation axis with a given angle.

    :param vector: The vector to be rotated given as [x, y, z].
    :param rotation_axis: The axis to be rotating around given as [x, y, z].
    :angle: The rotation angle in degree.
    """
    # Convert angle to radian.
    angle = np.deg2rad(angle)

    # Normalize the rotation_axis
    rotation_axis = map(float, rotation_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Use c1, c2, and c3 as shortcuts for the rotation axis.
    c1 = rotation_axis[0]
    c2 = rotation_axis[1]
    c3 = rotation_axis[2]

    # Build a column vector.
    vector = _get_vector(vector)

    # Build the rotation matrix.
    rotation_matrix = np.cos(angle) * \
        np.matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))) + \
        (1 - np.cos(angle)) * np.matrix(((c1 * c1, c1 * c2, c1 * c3),
                                         (c2 * c1, c2 * c2, c2 * c3),
                                         (c3 * c1, c3 * c2, c3 * c3))) + \
        np.sin(angle) * np.matrix(((0, -c3, c2), (c3, 0, -c1), (-c2, c1, 0)))

    # Rotate the vector.
    rotated_vector = np.array(rotation_matrix.dot(vector))

    # Make sure is also works for arrays of vectors.
    if rotated_vector.shape[0] > 1:
        return rotated_vector
    else:
        return rotated_vector.ravel()


def get_spherical_unit_vectors(lat, lon):
    """
    Returns the spherical unit vectors e_theta, e_phi and e_r for a point on
    the sphere determined by latitude and longitude which are defined as they
    are for earth.

    :param lat: Latitude in degree
    :param lon: Longitude in degree
    """
    colat = lat2colat(lat)
    # Convert to radian.
    colat, lon = map(np.deg2rad, [colat, lon])

    e_theta = _get_vector(np.cos(lon) * np.cos(colat),
                         np.sin(lon) * np.cos(colat),
                         -np.sin(colat))
    e_phi = _get_vector(-np.sin(lon), np.cos(lon), 0.0)
    e_r = _get_vector(np.cos(lon) * np.sin(colat),
                     np.sin(lon) * np.sin(colat),
                     np.cos(colat))
    return e_theta, e_phi, e_r


def rotate_lat_lon(lat, lon, rotation_axis, angle):
    """
    Takes a point specified by latitude and longitude and return a new pair of
    latitude longitude assuming the earth has been rotated around rotation_axis
    by angle.

    :param lat: Latitude of original point
    :param lon: Longitude of original point
    :rotation_axis: Rotation axis specified as [x, y, z].
    :angle: Rotation angle in degree.
    """
    # Convert to xyz. Do the calculation on the unit sphere as the radius does
    # not matter.
    xyz = lat_lon_radius_to_xyz(lat, lon, 1.0)
    # Rotate xyz.
    new_xyz = rotate_vector(xyz, rotation_axis, angle)
    new_lat, new_lon, _ = xyz_to_lat_lon_radius(new_xyz)
    return new_lat, new_lon


def xyz_to_lat_lon_radius(*args):
    """
    Converts x, y, and z to latitude, longitude and radius.

    >>> xyz_to_lat_lon_radius(1.0, 0.0, 0.0)
    (-0, 0.0, 1.0)

    >>> xyz_to_lat_lon_radius([1.0, 0.0, 0.0])
    (-0, 0.0, 1.0)

    >>> xyz_to_lat_lon_radius(0.0, 0.0, -2.0)
    (-90.0, 0.0, 2.0)

    >>> xyz_to_lat_lon_radius(0.0, 0.0, 2.0)
    (90.0, 0.0, 2.0)
    """
    xyz = _get_vector(*args)
    r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
    colat = np.arccos(xyz[2] / r)
    lon = np.arctan2(xyz[1], xyz[0])
    # Convert to degree.
    colat, lon = map(np.float32, map(np.rad2deg, [colat, lon]))
    lat = colat2lat(colat)
    return lat, lon, r


def lat_lon_radius_to_xyz(lat, lon, r):
    """
    Converts latitude, longitude and radius to x, y, and z.
    """
    colat = lat2colat(lat)
    # To radian
    colat, lon = map(np.deg2rad, [colat, lon])
    # Do the transformation
    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)
    return _get_vector(x, y, z)


def _get_rotation_and_base_transfer_matrix(lat, lon, rotation_axis, angle):
    """
    Generates a matrix that rotates a vector/tensor located at lat/lon and
    given in spherical coordinates for 'angle' degrees around 'rotation_axis'.
    Furthermore it performs a base change from the spherical unit vectors at
    lat/lon to the spherical unit vectors at the rotated lat/lon.

    This should never change the radial component.

    :param lat: Latitude of the recording point.
    :param lon: Longitude of the recording point.
    :param rotation_axis: Rotation axis given as [x, y, z].
    :param angle: Rotation angle in degree.

    >>> mat = _get_rotation_and_base_transfer_matrix(12, 34, [-45, 34, 45], -66)
    >>> mat[2, 2] - 1.0 <= 1E-7
    True
    >>> mat[2, 0] <= 1E-7
    True
    >>> mat[2, 1] <= 1E-7
    True
    >>> mat[0, 2] <= 1E-7
    True
    >>> mat[1, 2] <= 1E-7
    True
    """
    # Rotate latitude and longitude to obtain the new coordinates after the
    # rotation.
    lat_new, lon_new = rotate_lat_lon(lat, lon, rotation_axis, angle)

    # Get the orthonormal basis vectors at both points. This can be interpreted
    # as having two sets of basis vectors in the original xyz coordinate system.
    e_theta, e_phi, e_r = get_spherical_unit_vectors(lat, lon)
    e_theta_new, e_phi_new, e_r_new= get_spherical_unit_vectors(lat_new,
        lon_new)

    # Rotate the new unit vectors in the opposite direction to simulate a
    # rotation in the wanted direction.
    e_theta_new = rotate_vector(e_theta_new, rotation_axis, -angle)
    e_phi_new = rotate_vector(e_phi_new, rotation_axis, -angle)
    e_r_new = rotate_vector(e_r_new, rotation_axis, -angle)

    # Calculate the transfer matrix. This works because both sets of basis
    # vectors are orthonormal.
    transfer_matrix = np.matrix(( \
        [np.dot(e_theta_new, e_theta), np.dot(e_theta_new, e_phi), np.dot(e_theta_new, e_r)],
        [np.dot(e_phi_new, e_theta), np.dot(e_phi_new, e_phi), np.dot(e_phi_new, e_r)],
        [np.dot(e_r_new, e_theta), np.dot(e_r_new, e_phi), np.dot(e_r_new, e_r)]))
    return transfer_matrix


def rotate_moment_tensor(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, lat, lon, rotation_axis,
        angle):
    """
    Rotates a moment tensor, given in spherical coordinates, located at lat/lon
    around the rotation axis and simultaneously performs a base change from the
    spherical unit vectors at lat/lon to the unit vectors at the new
    coordinates.

    :param Mrr, Mtt, Mpp, Mrt, Mrp, Mtp: The six independent components of a
        moment tensor.
    :param lat: Latitude of the recording point.
    :param lon: Longitude of the recording point.
    :param rotation_axis: Rotation axis given as [x, y, z].
    :param angle: Rotation angle in degree.
    """
    transfer_matrix = _get_rotation_and_base_transfer_matrix(lat, lon,
        rotation_axis, angle)
    # Assemble the second order tensor.
    mt = np.matrix(([Mtt, Mtp, Mrt],
                    [Mtp, Mpp, Mrp],
                    [Mtp, Mrp, Mrr]))
    # Rotate it
    rotated_mt = transfer_matrix.dot(mt.dot(transfer_matrix.transpose()))
    # Return the six independent components in the same order as they were
    # given.
    return rotated_mt[2, 2], rotated_mt[0, 0], rotated_mt[1, 1], \
        rotated_mt[0, 2], rotated_mt[1, 2], rotated_mt[0, 1]


def rotate_data(north_data, east_data, vertical_data, lat, lon, rotation_axis,
    angle):
    """
    Rotates three component data recorded at lat/lon a certain amount of
    degrees around a given rotation axis.

    :param north_data: The north component of the data.
    :param east_data: The east component of the data.
    :param vertical_data: The vertical component of the data. Vertical is
        defined to be up, e.g. radially outwards.
    :param lat: Latitude of the recording point.
    :param lon: Longitude of the recording point.
    :param rotation_axis: Rotation axis given as [x, y, z].
    :param angle: Rotation angle in degree.
    """
    transfer_matrix = _get_rotation_and_base_transfer_matrix(lat, lon,
        rotation_axis, angle)

    # Apply the transfer matrix. Invert north data because they have
    # to point in the other direction to be consistent with the spherical
    # coordinates.
    new_data = np.array(transfer_matrix.dot([-1.0 * north_data,
        east_data, vertical_data]))

    # Return the transferred data arrays. Again negate north data.
    north_data = -1.0 * new_data[0]
    east_data = new_data[1]
    vertical_data = new_data[2]
    return north_data, east_data, vertical_data


####################################################################################
## Temporarily place the tests here.
####################################################################################

import unittest

class RotationsTestCase(unittest.TestCase):
    def test_rotate_vector(self):
        """
        Test the basic vector rotation around an arbitrary axis.
        """
        np.testing.assert_array_almost_equal( \
            rotate_vector([0, 0, 1], [0, 1, 0], 90),
            np.array([1.0, 0.0, 0.0]), decimal=10)
        np.testing.assert_array_almost_equal( \
            rotate_vector([0, 0, 1], [0, 1, 0], 180),
            np.array([0.0, 0.0, -1.0]), decimal=10)
        np.testing.assert_array_almost_equal( \
            rotate_vector([0, 0, 1], [0, 1, 0], 270),
            np.array([-1.0, 0.0, 0.0]), decimal=10)
        np.testing.assert_array_almost_equal( \
            rotate_vector([0, 0, 1], [0, 1, 0], 360),
            np.array([0.0, 0.0, 1.0]), decimal=10)
        np.testing.assert_array_almost_equal( \
            rotate_vector([0, 0, 1], [0, 1, 0], 45),
            np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5)]), decimal=10)
        # Use a different vector and rotation angle.
        np.testing.assert_array_almost_equal( \
            rotate_vector([1, 1, 1], [1, 0, 0], 90),
            np.array([1.0, -1.0, 1.0]), decimal=10)
        np.testing.assert_array_almost_equal( \
            rotate_vector([1, 1, 1], [1, 0, 0], 180),
            np.array([1.0, -1.0, -1.0]), decimal=10)
        np.testing.assert_array_almost_equal( \
            rotate_vector([1, 1, 1], [1, 0, 0], 270),
            np.array([1.0, 1.0, -1.0]), decimal=10)
        np.testing.assert_array_almost_equal( \
            rotate_vector([1, 1, 1], [1, 0, 0], 360),
            np.array([1.0, 1.0, 1.0]), decimal=10)
        # Some more arbitrary rotations.
        np.testing.assert_array_almost_equal( \
            rotate_vector([234.3, -5645.4, 345.45], [1, 0, 0], 360),
            np.array([234.3, -5645.4, 345.45]), decimal=10)
        np.testing.assert_array_almost_equal( \
            rotate_vector([234.3, -5645.4, 345.45], [1, 0, 0], 180),
            np.array([234.3, 5645.4, -345.45]), decimal=10)

    def test_get_spherical_unit_vectors(self):
        """
        Tests the get_spherical_unit_vectors() function.
        """
        # At 0, 0
        e_theta, e_phi, e_r = get_spherical_unit_vectors(0.0, 0.0)
        np.testing.assert_array_almost_equal( \
            e_theta, np.array([0, 0, -1]))
        np.testing.assert_array_almost_equal( \
            e_phi, np.array([0, 1, 0]))
        np.testing.assert_array_almost_equal( \
            e_r, np.array([1, 0, 0]))
        # At the north pole
        e_theta, e_phi, e_r = get_spherical_unit_vectors(90.0, 0.0)
        np.testing.assert_array_almost_equal( \
            e_theta, np.array([1, 0, 0]))
        np.testing.assert_array_almost_equal( \
            e_phi, np.array([0, 1, 0]))
        np.testing.assert_array_almost_equal( \
            e_r, np.array([0, 0, 1]))
        # At the south pole
        e_theta, e_phi, e_r = get_spherical_unit_vectors(-90.0, 0.0)
        np.testing.assert_array_almost_equal( \
            e_theta, np.array([-1, 0, 0]))
        np.testing.assert_array_almost_equal( \
            e_phi, np.array([0, 1, 0]))
        np.testing.assert_array_almost_equal( \
            e_r, np.array([0, 0, -1]))
        # At the "east pole"
        e_theta, e_phi, e_r = get_spherical_unit_vectors(0.0, 90.0)
        np.testing.assert_array_almost_equal( \
            e_theta, np.array([0, 0, -1]))
        np.testing.assert_array_almost_equal( \
            e_phi, np.array([-1, 0, 0]))
        np.testing.assert_array_almost_equal( \
            e_r, np.array([0, 1, 0]))
        # At the "west pole"
        e_theta, e_phi, e_r = get_spherical_unit_vectors(0.0, -90.0)
        np.testing.assert_array_almost_equal( \
            e_theta, np.array([0, 0, -1]))
        np.testing.assert_array_almost_equal( \
            e_phi, np.array([1, 0, 0]))
        np.testing.assert_array_almost_equal( \
            e_r, np.array([0, -1, 0]))

    def test_lat_lon_radius_to_xyz(self):
        """
        Test the lat_lon_radius_to_xyz() function.
        """
        # For (0/0)
        np.testing.assert_array_almost_equal( \
            lat_lon_radius_to_xyz(0.0, 0.0, 1.0),
            np.array([1.0, 0.0, 0.0]))
        # At the North Pole
        np.testing.assert_array_almost_equal( \
            lat_lon_radius_to_xyz(90.0, 0.0, 1.0),
            np.array([0.0, 0.0, 1.0]))
        # At the South Pole
        np.testing.assert_array_almost_equal( \
            lat_lon_radius_to_xyz(-90.0, 0.0, 1.0),
            np.array([0.0, 0.0, -1.0]))
        # At the "West Pole"
        np.testing.assert_array_almost_equal( \
            lat_lon_radius_to_xyz(0.0, -90.0, 1.0),
            np.array([0.0, -1.0, 0.0]))
        # At the "East Pole"
        np.testing.assert_array_almost_equal( \
            lat_lon_radius_to_xyz(0.0, 90.0, 1.0),
            np.array([0.0, 1.0, 0.0]))

    def test_xyz_to_lat_lon_radius(self):
        """
        Test the xyz_to_lat_lon_radius() function.
        """
        # For (0/0)
        lat, lon, radius = xyz_to_lat_lon_radius(1.0, 0.0, 0.0)
        self.assertAlmostEqual(lat, 0.0)
        self.assertAlmostEqual(lon, 0.0)
        self.assertAlmostEqual(radius, 1.0)
        # At the North Pole
        lat, lon, radius = xyz_to_lat_lon_radius(0.0, 0.0, 1.0)
        self.assertAlmostEqual(lat, 90.0)
        self.assertAlmostEqual(lon, 0.0)
        self.assertAlmostEqual(radius, 1.0)
        # At the South Pole
        lat, lon, radius = xyz_to_lat_lon_radius(0.0, 0.0, -1.0)
        self.assertAlmostEqual(lat, -90.0)
        self.assertAlmostEqual(lon, 0.0)
        self.assertAlmostEqual(radius, 1.0)
        # At the "West Pole"
        lat, lon, radius = xyz_to_lat_lon_radius(0.0, -1.0, 0.0)
        self.assertAlmostEqual(lat, 0.0)
        self.assertAlmostEqual(lon, -90.0)
        self.assertAlmostEqual(radius, 1.0)
        # At the "East Pole"
        lat, lon, radius = xyz_to_lat_lon_radius(0.0, 1.0, 0.0)
        self.assertAlmostEqual(lat, 0.0)
        self.assertAlmostEqual(lon, 90.0)
        self.assertAlmostEqual(radius, 1.0)

    def test_rotate_lat_lon(self):
        """
        Test the lat/lon rotation on a sphere.
        """
        # Rotate north pole to equator.
        lat_new, lon_new = rotate_lat_lon(90.0, 0.0, [0, 1, 0], 90)
        self.assertAlmostEqual(lat_new, 0.0)
        self.assertAlmostEqual(lon_new, 0.0)
        # Rotate north pole to the south pole.
        lat_new, lon_new = rotate_lat_lon(90.0, 0.0, [0, 1, 0], 180)
        self.assertAlmostEqual(lat_new, -90.0)
        self.assertAlmostEqual(lon_new, 0.0)
        # Rotate north pole to equator, the other way round.
        lat_new, lon_new = rotate_lat_lon(90.0, 0.0, [0, 1, 0], -90)
        self.assertAlmostEqual(lat_new, 0.0)
        self.assertAlmostEqual(lon_new, 180.0)
        # Rotate (0/0) to the east
        lat_new, lon_new = rotate_lat_lon(0.0, 0.0, [0, 0, 1], 90)
        self.assertAlmostEqual(lat_new, 0.0)
        self.assertAlmostEqual(lon_new, 90.0)
        # Rotate (0/0) to the west
        lat_new, lon_new = rotate_lat_lon(0.0, 0.0, [0, 0, 1], -90)
        self.assertAlmostEqual(lat_new, 0.0)
        self.assertAlmostEqual(lon_new, -90.0)
        # Rotate the west to the South Pole. The longitude can not be tested
        # reliably because is varies infinitly fast directly at a pole.
        lat_new, lon_new = rotate_lat_lon(0.0, -90.0, [1, 0, 0], 90)
        self.assertAlmostEqual(lat_new, -90.0)

    def test_rotate_data(self):
        """
        Test the rotate_data() function.
        """
        north_data = np.linspace(0, 10, 20)
        east_data = np.linspace(33, 44, 20)
        vertical_data = np.linspace(-12, -34, 20)
        # A rotation around the rotation axis of the earth with a source at the
        # equator should not change anything.
        new_north_data, new_east_data, new_vertical_data = \
            rotate_data(north_data, east_data, vertical_data, 0.0, 123.45,
                [0, 0, 1], 77.7)
        np.testing.assert_array_almost_equal(north_data, new_north_data, 5)
        np.testing.assert_array_almost_equal(east_data, new_east_data, 5)
        np.testing.assert_array_almost_equal(vertical_data, new_vertical_data, 5)
        # A rotation around the rotation axis of the earth should not change
        # the vertical component.
        new_north_data, new_east_data, new_vertical_data = \
            rotate_data(north_data, east_data, vertical_data, -55.66, 123.45,
                [0, 0, 1], 77.7)
        np.testing.assert_array_almost_equal(vertical_data, new_vertical_data, 5)
        # The same is true for any other rotation with an axis through the
        # center of the earth.
        new_north_data, new_east_data, new_vertical_data = \
            rotate_data(north_data, east_data, vertical_data, -55.66, 123.45,
                [123, 345.0, 0.234], 77.7)
        np.testing.assert_array_almost_equal(vertical_data, new_vertical_data, 5)
        # Any data along the Greenwich meridian and the opposite one should not
        # change with a rotation around the "East Pole" or the "West Pole".
        new_north_data, new_east_data, new_vertical_data = \
            rotate_data(north_data, east_data, vertical_data, 0.0, 0.0,
                [0, 1, 0], 55.0)
        np.testing.assert_array_almost_equal(north_data, new_north_data, 5)
        np.testing.assert_array_almost_equal(east_data, new_east_data, 5)
        np.testing.assert_array_almost_equal(vertical_data, new_vertical_data, 5)
        new_north_data, new_east_data, new_vertical_data = \
            rotate_data(north_data, east_data, vertical_data, 0.0, 0.0,
                [0, -1, 0], 55.0)
        np.testing.assert_array_almost_equal(north_data, new_north_data, 5)
        np.testing.assert_array_almost_equal(east_data, new_east_data, 5)
        np.testing.assert_array_almost_equal(vertical_data, new_vertical_data, 5)
        # A rotation of one hundred degree around the x-axis inverts (in this
        # case) north and east components.
        new_north_data, new_east_data, new_vertical_data = \
            rotate_data(north_data, east_data, vertical_data, 0.0, 90.0,
                [1, 0, 0], 180.0)
        np.testing.assert_array_almost_equal(north_data, -new_north_data, 5)
        np.testing.assert_array_almost_equal(east_data, -new_east_data, 5)
        np.testing.assert_array_almost_equal(vertical_data, new_vertical_data, 5)

    def test_rotate_moment_tensor(self):
        """
        Tests the moment tensor rotation.
        """
        # A full rotation should not change anything.
        Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotate_moment_tensor(1, 2, 3, 4,
            5, 6, 7, 8, [9, 10, 11], 360)
        self.assertAlmostEqual(Mrr, 1.0, 6)
        self.assertAlmostEqual(Mtt, 2.0, 6)
        self.assertAlmostEqual(Mpp, 3.0, 6)
        self.assertAlmostEqual(Mrt, 4.0, 6)
        self.assertAlmostEqual(Mrp, 5.0, 6)
        self.assertAlmostEqual(Mtp, 6.0, 6)

        # The following ones are tested against a well proven Matlab script by
        # Andreas Fichtner.
        Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotate_moment_tensor( \
            -0.704, 0.071, 0.632, 0.226, -0.611, 3.290,
            colat2lat(26.08), -21.17,
            [0, 1, 0], 57.5)
        Mrr_new, Mtt_new, Mpp_new, Mrt_new, Mrp_new, Mtp_new = \
            [-0.70400000, 2.04919171, -1.34619171, 0.02718681, -0.65089007,
            2.83207047]
        self.assertAlmostEqual(Mrr, Mrr_new, 6)
        self.assertAlmostEqual(Mtt, Mtt_new, 6)
        self.assertAlmostEqual(Mpp, Mpp_new, 6)
        self.assertAlmostEqual(Mrt, Mrt_new, 6)
        self.assertAlmostEqual(Mrp, Mrp_new, 6)
        self.assertAlmostEqual(Mtp, Mtp_new, 6)
        # Another example.
        Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotate_moment_tensor( \
            -0.818, -1.300, 2.120, 1.720, 2.290, -0.081,
            colat2lat(53.51), -9.87,
            [np.sqrt(2)/2, -np.sqrt(2)/2, 0], -31.34)
        Mrr_new, Mtt_new, Mpp_new, Mrt_new, Mrp_new, Mtp_new = \
            [-0.81800000, -0.69772178, 1.51772178, 2.55423451, 1.29552541,
            1.30522545]
        self.assertAlmostEqual(Mrr, Mrr_new, 6)
        self.assertAlmostEqual(Mtt, Mtt_new, 6)
        self.assertAlmostEqual(Mpp, Mpp_new, 6)
        self.assertAlmostEqual(Mrt, Mrt_new, 6)
        self.assertAlmostEqual(Mrp, Mrp_new, 6)
        self.assertAlmostEqual(Mtp, Mtp_new, 6)
        # The same as before, but with a non-normalized axis. Should work just
        # as well.
        Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotate_moment_tensor( \
            -0.818, -1.300, 2.120, 1.720, 2.290, -0.081,
            colat2lat(53.51), -9.87,
            [11.12, -11.12, 0], -31.34)
        Mrr_new, Mtt_new, Mpp_new, Mrt_new, Mrp_new, Mtp_new = \
            [-0.81800000, -0.69772178, 1.51772178, 2.55423451, 1.29552541,
            1.30522545]
        self.assertAlmostEqual(Mrr, Mrr_new, 6)
        self.assertAlmostEqual(Mtt, Mtt_new, 6)
        self.assertAlmostEqual(Mpp, Mpp_new, 6)
        self.assertAlmostEqual(Mrt, Mrt_new, 6)
        self.assertAlmostEqual(Mrp, Mrp_new, 6)
        self.assertAlmostEqual(Mtp, Mtp_new, 6)
        # One more.
        Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = rotate_moment_tensor( \
            0.952, -1.030, 0.076, 0.226, -0.040, -0.165,
            colat2lat(63.34), 55.80,
            [np.sqrt(3)/3, -np.sqrt(3)/3, -np.sqrt(3)/3], 123.45)
        Mrr_new, Mtt_new, Mpp_new, Mrt_new, Mrp_new, Mtp_new = \
            [0.95200000, -0.41458722, -0.53941278, -0.09170855, 0.21039378,
            0.57370606]
        self.assertAlmostEqual(Mrr, Mrr_new, 6)
        self.assertAlmostEqual(Mtt, Mtt_new, 6)
        self.assertAlmostEqual(Mpp, Mpp_new, 6)
        self.assertAlmostEqual(Mrt, Mrt_new, 6)
        self.assertAlmostEqual(Mrp, Mrp_new, 6)
        self.assertAlmostEqual(Mtp, Mtp_new, 6)


if __name__ == '__main__':
    # Run the doctests.
    import doctest
    # Run the unittests.
    doctest.testmod()
    unittest.main()
