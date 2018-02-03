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
    >>> vec # doctest: +NORMALIZE_WHITESPACE
    array([1., 2., 3.])
    >>> print(vec.dtype)
    float64

    >>> vec = _get_vector([1, 2, 3])
    >>> vec
    array([1., 2., 3.])
    >>> print(vec.dtype)
    float64

    >>> vec = _get_vector(np.array([1, 2, 3], dtype="int32"))
    >>> vec
    array([1., 2., 3.])
    >>> print(vec.dtype)
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
