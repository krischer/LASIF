#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization scripts.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

import rotations


def plot_domain(min_latitude, max_latitude, min_longitude, max_longitude,
    rotation_axis=[0.0, 0.0, 1.0], rotation_angle_in_degree=0.0,
    resolution="c"):

    bounds = rotations.get_max_extention_of_domain(min_latitude,
        max_latitude, min_longitude, max_longitude,
        rotation_axis=rotation_axis,
        rotation_angle_in_degree=rotation_angle_in_degree)
    center_lat = bounds["minimum_latitude"] + (bounds["maximum_latitude"] -
        bounds["minimum_latitude"]) / 2.0
    center_lng = bounds["minimum_longitude"] + (bounds["maximum_longitude"] -
        bounds["minimum_longitude"]) / 2.0

    m = Basemap(projection='ortho', lon_0=center_lng, lat_0=center_lat,
        resolution=resolution)
    m.drawmapboundary(fill_color='#cccccc')
    m.fillcontinents(color='white', lake_color='#cccccc', zorder=0)

    border = rotations.get_border_latlng_list(min_latitude, max_latitude,
        min_longitude, max_longitude)
    border = np.array(border)
    lats = border[:, 0]
    lngs = border[:, 1]
    lngs, lats = m(lngs, lats)
    m.plot(lngs, lats, color="red", lw=2, label="Simulation Domain")

    border = rotations.get_border_latlng_list(min_latitude, max_latitude,
        min_longitude, max_longitude, rotation_axis=rotation_axis,
        rotation_angle_in_degree=rotation_angle_in_degree)
    border = np.array(border)
    lats = border[:, 0]
    lngs = border[:, 1]
    lngs, lats = m(lngs, lats)
    m.plot(lngs, lats, color="black", lw=2, label="Physical Domain")
    plt.legend()

    plt.show()
