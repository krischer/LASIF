#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes handling the domain definition and associated functionality for LASIF.

Matplotlib is imported lazily to avoid heavy startup costs.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from abc import ABCMeta, abstractmethod
import numpy as np
from pyexodus import exodus
from scipy.spatial import cKDTree

from lasif.rotations import lat_lon_radius_to_xyz


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


class ExodusDomain(Domain):
    def __init__(self, exodus_file):
        self.e = exodus(exodus_file, mode='r')
        self.is_global_mesh = False
        self.domain_edge_tree = None
        self.earth_surface_tree = None
        self.approx_elem_width = None
        self.domain_edge_coords = None

        self._initialize_kd_trees()

    def _initialize_kd_trees(self):
        """
            Initialize two KDTrees to determine whether a point lies inside the domain
        """

        # if less than 2 side sets, this must be a global mesh, return
        if len(self.e.get_side_set_names()) < 2:
            self.is_global_mesh = True
            return

        side_nodes = []
        earth_surface_nodes = []
        for side_set in self.e.get_side_set_names():
            idx = self.e.get_side_set_ids()[
                        self.e.get_side_set_names().index(side_set)]

            if side_set == "r1":
                _, earth_surface_nodes = self.e.get_side_set_node_list(idx)
                continue

            _, nodes_side_set = list(self.e.get_side_set_node_list(idx))
            side_nodes.extend(nodes_side_set)

        # Remove Duplicates
        unique_nodes = np.unique(side_nodes)

        # Get node numbers of the nodes specifying the domain boundaries
        boundary_nodes = np.intersect1d(unique_nodes, earth_surface_nodes)

        # Deduct 1 from the nodes to get correct indices
        boundary_nodes -= 1
        earth_surface_nodes -= 1

        points = np.array(self.e.get_coords()).T
        self.domain_edge_coords = points[boundary_nodes]
        earth_surface_coords = points[earth_surface_nodes]

        # Get approximation of element width
        distances_to_node = self.domain_edge_coords - self.domain_edge_coords[0, :]
        r = np.sum(distances_to_node ** 2, axis=1) ** 0.5
        self.approx_elem_width = np.partition(r, 1)[1]

        # build KDTree that can be used for querying later
        self.earth_surface_tree = cKDTree(earth_surface_coords)
        self.domain_edge_tree = cKDTree(self.domain_edge_coords)

    def point_in_domain(self, longitude, latitude):
        if self.is_global_mesh:
            return True
        r_earth = 6371.0 * 1000.0
        point_on_surface = lat_lon_radius_to_xyz(latitude, longitude, r_earth)
        dist, _ = self.earth_surface_tree.query(point_on_surface, k=1)

        # False if not close to domain surface
        if dist > 2 * self.approx_elem_width:
            return False

        dist, _ = self.domain_edge_tree.query(point_on_surface, k=1)

        # False if too close to edge of domain
        if dist < 7 * self.approx_elem_width:
            return False
        return True

    def plot(self, plot_simulation_domain=False, ax=None):
        """
        Global domain is plotted using an equal area Mollweide projection.

        :param plot_simulation_domain: Parameter has no effect for the
            global domain.

        :return: The created GeoAxes instance.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap

        if ax is None:
            ax = plt.gca()
        plt.subplots_adjust(left=0.05, right=0.95)

        # Equal area mollweide projection.
        m = Basemap(projection='moll', lon_0=0, resolution="c", ax=ax)

        # Scatter plot domain edge nodes
        x, y, z = self.domain_edge_coords.T
        colats, lons, _ = np.degrees(cart2sph(x, y, z))
        lats = 90 - colats
        x, y = m(lons, lats)

        m.scatter(x, y, color='k')
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
        return "Exodus Domain"

def _plot_features(map_object, stepsize):
    """
    Helper function aiding in consistent plot styling.
    """
    import matplotlib.pyplot as plt

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
    import matplotlib.patheffects as PathEffects

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
