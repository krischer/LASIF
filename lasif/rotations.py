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

    :math:`\\theta` (theta) is the colatitude, e.g. 90.0 - latitude and is the
    angle from the z-axis. :math:`\phi` (phi) is the longitude and the angle
    from the x-axis towards the y-axis, a.k.a the azimuth angle. These are also
    the generally used spherical coordinates.

    All rotation axes have to be given as [x, y, z] in the just described
    coordinate system and all rotation angles have to given as degree. A
    positive rotation will rotate clockwise when looking in the direction of
    the rotation axis.

    For convenience reasons, most function in this module work with coordinates
    given in latitude and longitude.


:copyright: Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012-2013


:license: GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import math
import numpy as np
import sys

eps = sys.float_info.epsilon


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

    :param lat: The latitude.
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

    :param colat: The colatitude.
    """
    return -1.0 * (colat - 90.0)


def rotate_vector(vector, rotation_axis, angle):
    """
    Takes a vector and rotates it around a rotation axis with a given angle.

    :param vector: The vector to be rotated given as [x, y, z].
    :param rotation_axis: The axis to be rotating around given as [x, y, z].
    :param angle: The rotation angle in degree.
    """
    rotation_matrix = _get_rotation_matrix(rotation_axis, angle)

    # Build a column vector.
    vector = _get_vector(vector)

    # Rotate the vector.
    rotated_vector = np.array(rotation_matrix.dot(vector))

    # Make sure is also works for arrays of vectors.
    if rotated_vector.shape[0] > 1:
        return rotated_vector
    else:
        return rotated_vector.ravel()


def _get_rotation_matrix(axis, angle):
    """
    Returns the rotation matrix for the specified axis and angle.
    """
    axis = map(float, axis) / np.linalg.norm(axis)
    angle = np.deg2rad(angle)

    # Use c1, c2, and c3 as shortcuts for the rotation axis.
    c1 = axis[0]
    c2 = axis[1]
    c3 = axis[2]

    # Build the rotation matrix.
    rotation_matrix = np.cos(angle) * \
        np.matrix(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))) + \
        (1 - np.cos(angle)) * np.matrix(((c1 * c1, c1 * c2, c1 * c3),
                                         (c2 * c1, c2 * c2, c2 * c3),
                                         (c3 * c1, c3 * c2, c3 * c3))) + \
        np.sin(angle) * np.matrix(((0, -c3, c2), (c3, 0, -c1), (-c2, c1, 0)))
    return rotation_matrix


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
    :param rotation_axis: Rotation axis specified as [x, y, z].
    :param angle: Rotation angle in degree.
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
    (-0.0, 0.0, 1.0)

    >>> xyz_to_lat_lon_radius([1.0, 0.0, 0.0])
    (-0.0, 0.0, 1.0)

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

    :param lat: The latitude.
    :param lon:  The longitude.
    :param r: The radius.
    """
    colat = lat2colat(lat)
    # To radian
    colat, lon = map(np.deg2rad, [colat, lon])
    # Do the transformation
    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)
    return _get_vector(x, y, z)


def _get_axis_and_angle_from_rotation_matrix(M):
    """
    Extracts the axis and angle from a rotation matrix.

    >>> M = _get_rotation_matrix([0.26726124, 0.53452248, 0.80178373], 40)
    >>> _get_axis_and_angle_from_rotation_matrix(M)  \
    # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    (array([ 0.26726124,  0.53452248,  0.80178373]), 40.00...)
    """
    x = M[2, 1] - M[1, 2]
    y = M[0, 2] - M[2, 0]
    z = M[1, 0] - M[0, 1]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    t = M[0, 0] + M[1, 1] + M[2, 2]
    theta = np.arctan2(r, t - 1)
    axis = [x, y, z]
    axis /= np.linalg.norm(axis)

    return axis, np.rad2deg(theta)


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
    transfer_matrix = np.matrix((
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

    :param Mrr: A moment tensor component.
    :param Mtt: A moment tensor component.
    :param Mpp: A moment tensor component.
    :param Mrt: A moment tensor component.
    :param Mrp: A moment tensor component.
    :param Mtp: A moment tensor component.
    :param lat: Latitude of the recording point.
    :param lon: Longitude of the recording point.
    :param rotation_axis: Rotation axis given as [x, y, z].
    :param angle: Rotation angle in degree.
    """
    transfer_matrix = _get_rotation_and_base_transfer_matrix(
        lat, lon, rotation_axis, angle)
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
    transfer_matrix = _get_rotation_and_base_transfer_matrix(
        lat, lon, rotation_axis, angle)

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


def get_border_latlng_list(
        min_lat, max_lat, min_lng, max_lng, number_of_points_per_side=25,
        rotation_axis=(0, 0, 1), rotation_angle_in_degree=0):
    """
    Helper function taking a spherical section defined by latitudinal and
    longitudal extension, rotate it around the given axis and rotation angle
    and return a list of points outlining the region. Useful for plotting.

    :param min_lat: The minimum latitude.
    :param max_lat: The maximum latitude.
    :param min_lng: The minimum longitude.
    :param max_lng: The maximum longitude.
    :param number_of_points_per_side:  The number of points per side desired.
    :param rotation_axis: The rotation axis. Optional.
    :param rotation_angle_in_degree: The rotation angle in degrees. Optional.
    """
    north_border = np.empty((number_of_points_per_side, 2))
    south_border = np.empty((number_of_points_per_side, 2))
    east_border = np.empty((number_of_points_per_side, 2))
    west_border = np.empty((number_of_points_per_side, 2))

    north_border[:, 0] = max_lat
    north_border[:, 1] = np.linspace(min_lng, max_lng,
                                     number_of_points_per_side)

    east_border[:, 0] = np.linspace(max_lat, min_lat,
                                    number_of_points_per_side)
    east_border[:, 1] = max_lng

    south_border[:, 0] = min_lat
    south_border[:, 1] = np.linspace(max_lng, min_lng,
                                     number_of_points_per_side)

    west_border[:, 0] = np.linspace(min_lat, max_lat,
                                    number_of_points_per_side)
    west_border[:, 1] = min_lng

    # Rotate everything.
    for border in [north_border, south_border, east_border, west_border]:
        for _i in xrange(number_of_points_per_side):
            border[_i, 0], border[_i, 1] = rotate_lat_lon(
                border[_i, 0], border[_i, 1], rotation_axis,
                rotation_angle_in_degree)

    # Fix dateline wraparounds.
    for border in [north_border, south_border, east_border, west_border]:
        lngs = border[:, 1]
        lngs[lngs < min_lng] += 360.0

    # Take care to only use every corner once.
    borders = np.concatenate([north_border, east_border[1:], south_border[1:],
                              west_border[1:]])
    borders = list(borders)
    borders = [tuple(_i) for _i in borders]
    return borders


def get_center_angle(a, b):
    """
    Returns the angle of both angles on a sphere.

    :param a: Angle A in degrees.
    :param b: Angle B in degrees.

    The examples use round() to guard against floating point inaccuracies.

    >>> round(get_center_angle(350, 10), 9)
    0.0
    >>> round(get_center_angle(90, 270), 9)
    0.0
    >>> round(get_center_angle(-90, 90), 9)
    0.0
    >>> round(get_center_angle(350, 5), 9)
    357.5
    >>> round(get_center_angle(359, 10), 9)
    4.5
    >>> round(get_center_angle(10, 20), 9)
    15.0
    >>> round(get_center_angle(90, 180), 9)
    135.0
    >>> round(get_center_angle(0, 180), 9)
    90.0
    """
    a = math.radians(a)
    b = math.radians(b)

    a = (math.cos(a), math.sin(a))
    b = (math.cos(b), math.sin(b))

    c = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    if c[0] <= 10 * eps and c[1] <= 10 * eps:
        result = math.degrees((math.acos(a[0]) + math.pi / 2.0) % math.pi)
    else:
        result = math.degrees(math.atan2(c[1], c[0]))

    # Some predictability.
    if abs(result) <= (10.0 * eps):
        result = 0.0
    result %= 360.0

    return result


def get_max_extention_of_domain(min_lat, max_lat, min_lng, max_lng,
                                rotation_axis=(0, 0, 1),
                                rotation_angle_in_degree=0):
    """
    Helper function getting the maximum extends of a rotated domain.

    Returns a dictionary with the following keys:
        * minimum_latitude
        * maximum_latitude
        * minimum_longitude
        * maximum_longitude

    :param min_lat: The minimum latitude.
    :param max_lat: The maximum latitude.
    :param min_lng: The minimum longitude.
    :param max_lng: The maximum longitude.
    :param rotation_axis: The rotation axis in degree.
    :param rotation_angle_in_degree: The rotation angle in degree.
    """
    border = get_border_latlng_list(
        min_lat, max_lat, min_lng, max_lng, number_of_points_per_side=25,
        rotation_axis=rotation_axis,
        rotation_angle_in_degree=rotation_angle_in_degree)
    border = np.array(border)
    lats = border[:, 0]
    lngs = border[:, 1]
    return {
        "minimum_latitude": lats.min(),
        "maximum_latitude": lats.max(),
        "minimum_longitude": lngs.min(),
        "maximum_longitude": lngs.max()}


if __name__ == '__main__':
    import doctest
    doctest.testmod()
