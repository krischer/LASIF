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
GNU General Public License, Version 3
(http://www.gnu.org/copyleft/gpl.html)
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

    >>> mat = _get_rotation_and_base_transfer_matrix(12, 34, [-45, 34, 45], \
        -66)
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
    # as having two sets of basis vectors in the original xyz coordinate
    # system.
    e_theta, e_phi, e_r = get_spherical_unit_vectors(lat, lon)
    e_theta_new, e_phi_new, e_r_new = get_spherical_unit_vectors(lat_new,
        lon_new)

    # Rotate the new unit vectors in the opposite direction to simulate a
    # rotation in the wanted direction.
    e_theta_new = rotate_vector(e_theta_new, rotation_axis, -angle)
    e_phi_new = rotate_vector(e_phi_new, rotation_axis, -angle)
    e_r_new = rotate_vector(e_r_new, rotation_axis, -angle)

    # Calculate the transfer matrix. This works because both sets of basis
    # vectors are orthonormal.
    transfer_matrix = np.matrix(( \
        [np.dot(e_theta_new, e_theta), np.dot(e_theta_new, e_phi),
            np.dot(e_theta_new, e_r)],
        [np.dot(e_phi_new, e_theta), np.dot(e_phi_new, e_phi),
            np.dot(e_phi_new, e_r)],
        [np.dot(e_r_new, e_theta), np.dot(e_r_new, e_phi), np.dot(e_r_new,
            e_r)]))
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
