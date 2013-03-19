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
from obspy.imaging.beachball import Beach

import rotations


def plot_domain(min_latitude, max_latitude, min_longitude, max_longitude,
        boundary_buffer_in_degree=0.0, rotation_axis=[0.0, 0.0, 1.0],
        rotation_angle_in_degree=0.0, resolution="c", show_plot=True,
        plot_simulation_domain=False):
    """
    """
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
        min_longitude, max_longitude, rotation_axis=rotation_axis,
        rotation_angle_in_degree=rotation_angle_in_degree)
    border = np.array(border)
    lats = border[:, 0]
    lngs = border[:, 1]
    lngs, lats = m(lngs, lats)
    m.plot(lngs, lats, color="black", lw=2, label="Physical Domain")

    if boundary_buffer_in_degree:
        border = rotations.get_border_latlng_list(
            min_latitude + boundary_buffer_in_degree,
            max_latitude - boundary_buffer_in_degree,
            min_longitude + boundary_buffer_in_degree,
            max_longitude - boundary_buffer_in_degree,
            rotation_axis=rotation_axis,
            rotation_angle_in_degree=rotation_angle_in_degree)
        border = np.array(border)
        lats = border[:, 0]
        lngs = border[:, 1]
        lngs, lats = m(lngs, lats)
        m.plot(lngs, lats, color="black", lw=2, alpha=0.4)

    if plot_simulation_domain is True:
        border = rotations.get_border_latlng_list(min_latitude, max_latitude,
            min_longitude, max_longitude)
        border = np.array(border)
        lats = border[:, 0]
        lngs = border[:, 1]
        lngs, lats = m(lngs, lats)
        m.plot(lngs, lats, color="red", lw=2, label="Simulation Domain")

        if boundary_buffer_in_degree:
            border = rotations.get_border_latlng_list(
                min_latitude + boundary_buffer_in_degree,
                max_latitude - boundary_buffer_in_degree,
                min_longitude + boundary_buffer_in_degree,
                max_longitude - boundary_buffer_in_degree)
            border = np.array(border)
            lats = border[:, 0]
            lngs = border[:, 1]
            lngs, lats = m(lngs, lats)
            m.plot(lngs, lats, color="red", lw=2, alpha=0.4)
        plt.legend()

    if show_plot is True:
        plt.show()

    return m


def plot_events(events, map_object):
    """
    """
    for event in events:
        org = event.preferred_origin() or event.origins[0]
        fm = event.preferred_focal_mechanism() or event.focal_mechanisms[0]
        mt = fm.moment_tensor.tensor

        # Add beachball plot.
        x, y = map_object(org.longitude, org.latitude)

        focmec = [mt.m_rr, mt.m_tt, mt.m_pp, mt.m_rt, mt.m_rp, mt.m_tp]
        # Attempt to calculate the best beachball size.
        width = max((map_object.xmax - map_object.xmin,
            map_object.ymax - map_object.ymin)) * 0.040
        b = Beach(focmec, xy=(x, y), width=width, linewidth=1, facecolor="red")
        b.set_zorder(200000000)
        plt.gca().add_collection(b)
