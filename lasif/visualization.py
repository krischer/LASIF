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
from itertools import izip
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from obspy.imaging.beachball import Beach
from obspy.signal.tf_misfit import plotTfr

import rotations


def plot_domain(min_latitude, max_latitude, min_longitude, max_longitude,
        boundary_buffer_in_degree=0.0, rotation_axis=[0.0, 0.0, 1.0],
        rotation_angle_in_degree=0.0, show_plot=True,
        plot_simulation_domain=False, zoom=False):
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

    extend_x = bounds["maximum_longitude"] - bounds["minimum_longitude"]
    extend_y = bounds["maximum_latitude"] - bounds["minimum_latitude"]
    max_extend = max(extend_x, extend_y)

    # Arbitrary threshold
    if zoom is False or max_extend > 70:
        m = Basemap(projection='ortho', lon_0=center_lng, lat_0=center_lat,
            resolution="c")
    else:
        buffer = max_extend * 0.1
        m = Basemap(projection='merc', resolution="l",
            llcrnrlat=bounds["minimum_latitude"] - buffer,
            urcrnrlat=bounds["maximum_latitude"] + buffer,
            llcrnrlon=bounds["minimum_longitude"] - buffer,
            urcrnrlon=bounds["maximum_longitude"] + buffer)

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
            map_object.ymax - map_object.ymin)) * 0.020
        b = Beach(focmec, xy=(x, y), width=width, linewidth=1, facecolor="red")
        b.set_zorder(200000000)
        plt.gca().add_collection(b)


def plot_raydensity(map_object, station_events, min_lat, max_lat, min_lng,
        max_lng, rot_axis, rot_angle):
    """
    Create a ray-density plot for all events and all stations.
    """
    from lasif.tools.great_circle_binner import GreatCircleBinner, Point
    import progressbar

    bounds = rotations.get_max_extention_of_domain(min_lat, max_lat, min_lng,
        max_lng, rotation_axis=rot_axis, rotation_angle_in_degree=rot_angle)

    binner = GreatCircleBinner(bounds["minimum_latitude"],
        bounds["maximum_latitude"], 3000, bounds["minimum_longitude"],
        bounds["maximum_longitude"], 3000)

    station_count = sum([len(_i[1]) for _i in station_events])

    widgets = ["Calculating greatcircles: ", progressbar.Percentage(),
        progressbar.Bar(), "", progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets,
        maxval=station_count).start()

    _i = 0
    for event, stations in station_events:
        org = event.preferred_origin() or event.origins[0]
        e_point = Point(org.latitude, org.longitude)
        for station in stations.itervalues():
            _i += 1
            pbar.update(_i)
            binner.add_greatcircle(e_point, Point(station["latitude"],
                station["longitude"]))
    pbar.finish()

    lngs, lats = binner.coordinates

    cmap = cm.get_cmap("gist_heat_r")
    cmap._init()
    cmap._lut[:20, -1] = np.linspace(0, 1.0, 20)
    ln, la = map_object(lngs, lats)
    map_object.pcolormesh(ln, la, binner.bins.transpose(), cmap=cmap)


def plot_stations_for_event(map_object, station_dict, event_info):
    """
    Plots all stations for one event.

    :param station_dict: A dictionary whose values at least contain latitude
        and longitude keys.
    """
    # Plot the stations with scatter.
    lngs = [_i["longitude"] for _i in station_dict.itervalues()]
    lats = [_i["latitude"] for _i in station_dict.itervalues()]
    x, y = map_object(lngs, lats)
    map_object.scatter(x, y, color="green", s=35, marker="v", zorder=100,
        edgecolor="black")

    # Plot the ray paths.
    for sta_lng, sta_lat in izip(lngs, lats):
        map_object.drawgreatcircle(event_info["longitude"],
            event_info["latitude"], sta_lng, sta_lat, lw=2, alpha=0.3)

    title = "Event in %s, at %s, %.1f Mw, with %i stations." % (
        event_info["region"], str(event_info["origin_time"]),
        event_info["magnitude"], len(station_dict))
    plt.gca().set_title(title, size="large")


def plot_tf(data, delta):
    """
    Plots a time frequency representation of any time series.
    """
    npts = len(data)

    plotTfr(data, dt=delta, fmin=1.0 / (npts * delta),
        fmax=1.0 / (2.0 * delta))
