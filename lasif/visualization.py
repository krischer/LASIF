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
from obspy.imaging.mopad_wrapper import Beach
from obspy.signal.tf_misfit import plotTfr
import os

import rotations


def plot_domain(min_latitude, max_latitude, min_longitude, max_longitude,
        boundary_buffer_in_degree=0.0, rotation_axis=[0.0, 0.0, 1.0],
        rotation_angle_in_degree=0.0, show_plot=True,
        plot_simulation_domain=False, zoom=False, resolution=None):
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
        if resolution is None:
            resolution = "c"
        m = Basemap(projection='ortho', lon_0=center_lng, lat_0=center_lat,
            resolution=resolution)
    else:
        if resolution is None:
            resolution = "l"
        buffer = max_extend * 0.1
        m = Basemap(projection='merc', resolution=resolution,
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
    def event_pick_handler(event, annotations=[]):
        # Remove any potentially existing annotations.
        for i in annotations:
            i.remove()
        annotations[:] = []
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        annotation = plt.annotate(event.artist.detailed_description,
            xy=(x, y), xytext=(0.98, 0.98), textcoords="figure fraction",
            horizontalalignment="right", verticalalignment="top",
            arrowprops=dict(arrowstyle="fancy", color="0.5",
            connectionstyle="arc3,rad=0.3"), zorder=10E9,
            fontsize="small")
        annotations.append(annotation)
        plt.gcf().canvas.draw()

    for event in events:
        org = event.preferred_origin() or event.origins[0]
        mag = event.preferred_magnitude() or event.magnitudes[0]
        fm = event.preferred_focal_mechanism() or event.focal_mechanisms[0]
        mt = fm.moment_tensor.tensor

        # Add beachball plot.
        x, y = map_object(org.longitude, org.latitude)

        focmec = [mt.m_rr, mt.m_tt, mt.m_pp, mt.m_rt, mt.m_rp, mt.m_tp]
        # Attempt to calculate the best beachball size.
        width = max((map_object.xmax - map_object.xmin,
            map_object.ymax - map_object.ymin)) * 0.020
        b = Beach(focmec, xy=(x, y), width=width, linewidth=1, facecolor="red")
        b.set_picker(True)
        b.detailed_description = (
            "Event %.1f %s\n"
            "Lat: %.1f, Lng: %.1f, Depth: %.1f km\n"
            "Time: %s\n"
            "%s"
        ) % (mag.mag, mag.magnitude_type, org.latitude, org.longitude,
            org.depth / 1000.0, org.time, os.path.basename(event.filename))

        b.set_zorder(200000000)
        plt.gca().add_collection(b)

    plt.gcf().canvas.mpl_connect("pick_event", event_pick_handler)


def plot_raydensity(map_object, station_events, min_lat, max_lat, min_lng,
        max_lng, rot_axis, rot_angle):
    """
    Create a ray-density plot for all events and all stations.

    This function is potentially expensive and will use all CPUs available.
    Does require geographiclib to be installed.
    """
    import ctypes as C
    from lasif.tools.great_circle_binner import GreatCircleBinner, Point
    import multiprocessing
    import progressbar
    from scipy.stats import scoreatpercentile

    bounds = rotations.get_max_extention_of_domain(min_lat, max_lat, min_lng,
        max_lng, rotation_axis=rot_axis, rotation_angle_in_degree=rot_angle)

    # Merge everything so that a list with coordinate pairs is created. This
    # list is then distributed among all processors.
    station_event_list = []
    for event, stations in station_events:
        org = event.preferred_origin() or event.origins[0]
        e_point = Point(org.latitude, org.longitude)
        for station in stations.itervalues():
            station_event_list.append((e_point, Point(station["latitude"],
                station["longitude"])))

    circle_count = len(station_event_list)

    # The granularity of the latitude/longitude discretization for the
    # raypaths. Attempt to get a somewhat meaningful result in any case.
    lat_lng_count = 1000
    if circle_count < 1000:
        lat_lng_count = 1000
    if circle_count < 10000:
        lat_lng_count = 2000
    else:
        lat_lng_count = 3000

    cpu_count = multiprocessing.cpu_count()

    def to_numpy(raw_array, dtype, shape):
        data = np.frombuffer(raw_array.get_obj())
        data.dtype = dtype
        return data.reshape(shape)

    print "\nLaunching %i greatcircle calculations on %i CPUs..." % \
        (circle_count, cpu_count)

    widgets = ["Progress: ", progressbar.Percentage(),
        progressbar.Bar(), "", progressbar.ETA()]
    pbar = progressbar.ProgressBar(widgets=widgets,
        maxval=circle_count).start()

    def great_circle_binning(sta_evs, bin_data_buffer, bin_data_shape,
            lock, counter):
        new_bins = GreatCircleBinner(bounds["minimum_latitude"],
            bounds["maximum_latitude"], lat_lng_count,
            bounds["minimum_longitude"], bounds["maximum_longitude"],
            lat_lng_count)
        for event, station in sta_evs:
            with lock:
                counter.value += 1
            if not counter.value % 25:
                pbar.update(counter.value)
            new_bins.add_greatcircle(event, station)

        bin_data = to_numpy(bin_data_buffer, np.uint32, bin_data_shape)
        with bin_data_buffer.get_lock():
            bin_data += new_bins.bins

    # Split the data in cpu_count parts.
    def chunk(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out
    chunks = chunk(station_event_list, cpu_count)

    # One instance that collects everything.
    collected_bins = GreatCircleBinner(bounds["minimum_latitude"],
        bounds["maximum_latitude"], lat_lng_count, bounds["minimum_longitude"],
        bounds["maximum_longitude"], lat_lng_count)

    # Use a multiprocessing shared memory array and map it to a numpy view.
    collected_bins_data = multiprocessing.Array(C.c_uint32,
        collected_bins.bins.size)
    collected_bins.bins = to_numpy(collected_bins_data, np.uint32,
        collected_bins.bins.shape)

    # Create, launch and join one process per CPU. Use a shared value as a
    # counter and a lock to avoid race conditions.
    processes = []
    lock = multiprocessing.Lock()
    counter = multiprocessing.Value("i", 0)
    for _i in xrange(cpu_count):
        processes.append(multiprocessing.Process(target=great_circle_binning,
            args=(chunks[_i], collected_bins_data, collected_bins.bins.shape,
                lock, counter)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    pbar.finish()

    title = "%i Events with %i recorded 3 component waveforms" % (
        len(station_events), circle_count)
    #plt.gca().set_title(title, size="large")
    plt.title(title, size="xx-large")

    data = collected_bins.bins.transpose()

    if data.max() >= 10:
        data = np.log10(data)
        data += 0.1
        data[np.isinf(data)] = 0.0
        max_val = scoreatpercentile(data.ravel(), 99)
    else:
        max_val = data.max()

    cmap = cm.get_cmap("gist_heat")
    cmap._init()
    cmap._lut[:120, -1] = np.linspace(0, 1.0, 120) ** 2

    # Slightly change the appearance of the map so it suits the rays.
    map_object.drawmapboundary(fill_color='#bbbbbb')
    map_object.fillcontinents(color='#dddddd', lake_color='#dddddd', zorder=0)

    lngs, lats = collected_bins.coordinates
    ln, la = map_object(lngs, lats)
    map_object.pcolormesh(ln, la, data, cmap=cmap, vmin=0, vmax=max_val)
    # Draw the coastlines so they appear over the rays. Otherwise things are
    # sometimes hard to see.
    map_object.drawcoastlines()
    map_object.drawcountries(linewidth=0.2)
    map_object.drawmeridians(np.arange(0, 360, 30))
    map_object.drawparallels(np.arange(-90, 90, 30))


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


def plot_event_histogram(events, plot_type):
    from matplotlib.dates import date2num, num2date
    from matplotlib import ticker

    plt.figure(figsize=(12, 4))

    values = []
    for event in events:
        org = event.preferred_origin() or event.origins[0]
        if plot_type == "depth":
            values.append(org.depth / 1000.0)
        elif plot_type == "time":
            values.append(date2num(org.time.datetime))

    plt.hist(values, bins=250)

    if plot_type == "time":
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda numdate, _: num2date(numdate).strftime('%Y-%d-%m')))
        plt.gcf().autofmt_xdate()
        plt.xlabel("Origin time (UTC)")
        plt.title("Origin time distribution (%i events)" % len(events))
    elif plot_type == "depth":
        plt.xlabel("Event depth in km")
        plt.title("Hypocenter depth distribution (%i events)" % len(events))

    plt.tight_layout()
