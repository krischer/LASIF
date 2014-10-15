#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes handling the domain definition and associated functionality for LASIF.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from abc import ABCMeta, abstractmethod

import collections
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import math
import numpy as np

from lasif import rotations


class Domain(object):
    """
    Abstract base class for the domain definitions.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def point_in_domain(self, longitude, latitude):
        """
        Called to determine if a certain point is contained by the domain.

        :param longitude: The longitude of the point.
        :param latitude: The latitude of the point.
        :return: bool
        """
        pass

    @abstractmethod
    def plot(self, plot_simulation_domain=False, ax=None):
        """
        Plots the domain and attempts to choose a reasonable projection for
        all possible settings. Will likely break for some settings.

        :param plot_simulation_domain: Parameter has no effect for the
            global domain.

        :return: The created GeoAxes instance.
        """
        pass

    @abstractmethod
    def get_max_extent(self):
        """
        Returns the maximum extends of the domain.

        Returns a dictionary with the following keys:
            * minimum_latitude
            * maximum_latitude
            * minimum_longitude
            * maximum_longitude
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)


class RectangularSphericalSection(Domain):
    """
    Class defining a potentially rotated spherical section.
    """
    def __init__(self, min_longitude, max_longitude, min_latitude,
                 max_latitude, min_depth_in_km=0.0, max_depth_in_km=6371.0,
                 rotation_axis=[0.0, 0.0, 1.0],
                 rotation_angle_in_degree=0.0,
                 boundary_width_in_degree=0.0):
        self.min_longitude = min_longitude
        self.max_longitude = max_longitude
        self.min_latitude = min_latitude
        self.max_latitude = max_latitude
        self.min_depth_in_km = min_depth_in_km
        self.max_depth_in_km = max_depth_in_km
        self.rotation_axis = rotation_axis
        self.rotation_angle_in_degree = rotation_angle_in_degree
        self.boundary_width_in_degree = boundary_width_in_degree

    @property
    def border(self):
        return rotations.get_border_latlng_list(
            min_lat=self.min_latitude, max_lat=self.max_latitude,
            min_lng=self.min_longitude, max_lng=self.max_longitude,
            number_of_points_per_side=50, rotation_axis=self.rotation_axis,
            rotation_angle_in_degree=self.rotation_angle_in_degree)

    @property
    def inner_border(self):
        return rotations.get_border_latlng_list(
            min_lat=self.min_latitude + self.boundary_width_in_degree,
            max_lat=self.max_latitude - self.boundary_width_in_degree,
            min_lng=self.min_longitude + self.boundary_width_in_degree,
            max_lng=self.max_longitude - self.boundary_width_in_degree,
            number_of_points_per_side=50, rotation_axis=self.rotation_axis,
            rotation_angle_in_degree=self.rotation_angle_in_degree)

    @property
    def unrotated_border(self):
        return rotations.get_border_latlng_list(
            min_lat=self.min_latitude, max_lat=self.max_latitude,
            min_lng=self.min_longitude, max_lng=self.max_longitude,
            number_of_points_per_side=50)

    @property
    def unrotated_inner_border(self):
        return rotations.get_border_latlng_list(
            min_lat=self.min_latitude + self.boundary_width_in_degree,
            max_lat=self.max_latitude - self.boundary_width_in_degree,
            min_lng=self.min_longitude + self.boundary_width_in_degree,
            max_lng=self.max_longitude - self.boundary_width_in_degree,
            number_of_points_per_side=50)

    @property
    def center(self):
        """
        Get the center of the domain.
        """
        c = self.unrotated_center
        Point = collections.namedtuple("CenterPoint", ["longitude",
                                                       "latitude"])
        r_lat, r_lng = rotations.rotate_lat_lon(
            c.latitude, c.longitude, self.rotation_axis,
            self.rotation_angle_in_degree)
        return Point(longitude=r_lng, latitude=r_lat)

    @property
    def unrotated_center(self):
        c_lng = rotations.get_center_angle(self.max_longitude,
                                           self.min_longitude)
        c_lat = rotations.colat2lat(rotations.get_center_angle(
            rotations.lat2colat(self.max_latitude),
            rotations.lat2colat(self.min_latitude)))

        Point = collections.namedtuple("CenterPoint", ["longitude",
                                                       "latitude"])
        return Point(longitude=c_lng, latitude=c_lat)

    @property
    def max_extent(self):
        # Assume worst case of square domain.
        return max(self.max_latitude - self.min_latitude,
                   self.max_longitude - self.min_longitude) * math.sqrt(2.0)

    @property
    def extent(self):
        lats, lngs = zip(*self.border)
        Extent = collections.namedtuple(
            "Extent", ["max_latitude", "min_latitude", "max_longitude",
                       "min_longitude", "latitudinal_extent",
                       "longitudinal_extent"])
        return Extent(max_latitude=max(lats),
                      min_latitude=min(lats),
                      max_longitude=max(lngs),
                      min_longitude=min(lngs),
                      latitudinal_extent=abs(max(lats) - min(lats)),
                      longitudinal_extent=abs(max(lngs) - min(lngs)))

    def point_in_domain(self, longitude, latitude):
        """
        Checks if a geographic point is placed inside a rotated spherical
        section. It simple rotates the point and checks if it is inside the
        unrotated domain. It therefore works for all possible projections
        and what not.

        :param longitude: The longitude of the point.
        :param latitude:  The latitude of the point.

        :return: bool
        """
        if self.rotation_angle_in_degree:
            # Rotate the point.
            r_lat, r_lng = rotations.rotate_lat_lon(
                latitude, longitude, self.rotation_axis,
                -1.0 * self.rotation_angle_in_degree)
        else:
            r_lng = longitude
            r_lat = latitude

        bw = self.boundary_width_in_degree

        # Check if in bounds.
        if not ((self.min_latitude + bw) <= r_lat <=
                (self.max_latitude - bw)) or \
                not ((self.min_longitude + bw) <= r_lng <=
                     (self.max_longitude - bw)):
            return False

        return True

    def plot(self, plot_simulation_domain=False, ax=None):
        plt.subplots_adjust(left=0.05, right=0.95)

        # Use a global plot for very large domains.
        if self.max_extent >= 180.0:
            m = Basemap(projection='moll', lon_0=0, resolution="c", ax=ax)
            stepsize = 45.0
        # Orthographic projection for 75.0 <= extent < 180.0
        elif self.max_extent >= 75.0 or (plot_simulation_domain is True and
                                         self.rotation_angle_in_degree):
            m = Basemap(projection="ortho", lon_0=self.center.longitude,
                        lat_0=self.center.latitude, resolution="c", ax=ax)
            stepsize = 10.0
        # Lambert azimuthal equal area projection. Equal area projections
        # are useful for interpreting features and this particular one also
        # does not distort features a lot on regional scales.
        else:
            extent = self.extent
            # Calculate approximate width and height in meters.
            width = extent.longitudinal_extent
            height = extent.latitudinal_extent

            if width > 50.0:
                stepsize = 10.0
            elif 20.0 < width <= 50.0:
                stepsize = 5.0
            elif 5.0 < width <= 20.0:
                stepsize = 2.0
            else:
                stepsize = 1.0

            width *= 110000 * 1.1
            height *= 110000 * 1.3

            m = Basemap(projection='laea', resolution="l", width=width,
                        height=height, lat_0=self.center.latitude,
                        lon_0=self.center.longitude, ax=ax)

        _plot_features(m, stepsize)

        if plot_simulation_domain and self.rotation_angle_in_degree:
            _plot_lines(m, self.unrotated_border, color="red", lw=2,
                        label="Simulation Domain")
            if self.boundary_width_in_degree:
                _plot_lines(m, self.unrotated_inner_border, color="red", lw=2,
                            alpha=0.4)

        _plot_lines(m, self.border, color="black", lw=2,
                    label="Physical Domain", effects=True)
        if self.boundary_width_in_degree:
            _plot_lines(m, self.inner_border, color="black", lw=2, alpha=0.4,
                        effects=True)

        if plot_simulation_domain and self.rotation_angle_in_degree:
            plt.legend(framealpha=0.5)

        return m

    def get_max_extent(self):
        """
        Returns the maximum extends of the domain.

        Returns a dictionary with the following keys:
            * minimum_latitude
            * maximum_latitude
            * minimum_longitude
            * maximum_longitude
        """
        return rotations.get_max_extention_of_domain(
            self.min_latitude, self.max_latitude, self.min_longitude,
            self.max_longitude,
            rotation_axis=self.rotation_axis,
            rotation_angle_in_degree=self.rotation_angle_in_degree)

    def __str__(self):
        ret_str = (
            "{rotation} Spherical Section Domain\n"
            u"\tLatitude: {min_lat:.2f}° - {max_lat:.2f}°\n"
            u"\tLongitude: {min_lng:.2f}° - {max_lng:.2f}°\n"
            "\tDepth: {min_depth:.1f}km - {max_depth:.1f}km"
        )
        if self.rotation_angle_in_degree:
            rotation = "Rotated"
            ret_str += (
                "\n\tRotation Axis: {x:.1f} / {y:.1f} / {z:.1f}\n"
                u"\tRotation Angle: {angle:.2f}°"
            )
        else:
            rotation = "Unrotated"
        return ret_str.format(
            rotation=rotation,
            min_lat=self.min_latitude,
            max_lat=self.max_latitude,
            min_lng=self.min_longitude,
            max_lng=self.max_longitude,
            min_depth=self.min_depth_in_km,
            max_depth=self.max_depth_in_km,
            x=self.rotation_axis[0],
            y=self.rotation_axis[1],
            z=self.rotation_axis[2],
            angle=self.rotation_angle_in_degree).encode("utf-8")


class GlobalDomain(Domain):
    def point_in_domain(self, longitude, latitude):
        """
        Naturally contains every point and always returns True.

        :param longitude: The longitude of the point.
        :param latitude: The latitude of the point.

        :return: bool
        """
        return True

    def plot(self, plot_simulation_domain=False, ax=None):
        """
        Global domain is plotted using an equal area Mollweide projection.

        :param plot_simulation_domain: Parameter has no effect for the
            global domain.

        :return: The created GeoAxes instance.
        """
        plt.subplots_adjust(left=0.05, right=0.95)

        # Equal area mollweide projection.
        m = Basemap(projection='moll', lon_0=0, resolution="c", ax=ax)
        _plot_features(m, stepsize=45)
        return m

    def get_max_extent(self):
        """
        Returns the maximum extends of the domain.

        Returns a dictionary with the following keys:
            * minimum_latitude
            * maximum_latitude
            * minimum_longitude
            * maximum_longitude
        """
        return {"minimum_latitude": -90.0, "maximum_latitude": 90.0,
                "minimum_longitude": -180.0, "maximum_longitude": 180.0}

    def __str__(self):
        return "Global Domain"


def _plot_features(map_object, stepsize):
    """
    Helper function aiding in consistent plot styling.
    """
    map_object.drawmapboundary(fill_color='#bbbbbb')
    map_object.fillcontinents(color='white', lake_color='#cccccc', zorder=0)
    plt.gcf().patch.set_alpha(0.0)

    # Style for parallels and meridians.
    LINESTYLE = {
        "linewidth": 0.5,
        "dashes": [],
        "color": "#999999"}

    # Parallels.
    if map_object.projection in ["moll", "laea"]:
        label = True
    else:
        label = False
    parallels = np.arange(-90.0, 90.0, stepsize)
    map_object.drawparallels(parallels, labels=[False, label, False, False],
                             zorder=200, **LINESTYLE)
    # Meridians.
    if map_object.projection in ["laea"]:
        label = True
    else:
        label = False
    meridians = np.arange(0.0, 360.0, stepsize)
    map_object.drawmeridians(
        meridians, labels=[False, False, False, label], zorder=200,
        **LINESTYLE)

    map_object.drawcoastlines(color="#444444", linewidth=0.7)
    map_object.drawcountries(linewidth=0.2, color="#999999")


def _plot_lines(map_object, lines, color, lw, alpha=1.0, label=None,
                effects=False):
    lines = np.array(lines)
    lats = lines[:, 0]
    lngs = lines[:, 1]
    lngs, lats = map_object(lngs, lats)

    # Fix to avoid deal with basemaps inability to plot lines across map
    # boundaries.
    # XXX: No local area stitching so far!
    if map_object.projection == "ortho":
        lats = np.ma.masked_greater(lats, 1E15)
        lngs = np.ma.masked_greater(lngs, 1E15)
    elif map_object.projection == "moll":
        x = np.diff(lngs)
        y = np.diff(lats)
        lats = np.ma.array(lats, mask=False)
        lngs = np.ma.array(lngs, mask=False)
        max_jump = 0.3 * min(
            map_object.xmax - map_object.xmin,
            map_object.ymax - map_object.ymin)
        idx_1 = np.where(np.abs(x) > max_jump)
        idx_2 = np.where(np.abs(y) > max_jump)
        if idx_1:
            lats.mask[idx_1] = True
        if idx_2:
            lats.mask[idx_2] = True
        lngs.mask = lats.mask

    path_effects = [PathEffects.withStroke(linewidth=5, foreground="white")] \
        if effects else None

    map_object.plot(lngs, lats, color=color, lw=lw, alpha=alpha,
                    label=label, path_effects=path_effects)
