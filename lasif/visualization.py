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
import numpy as np
from obspy.imaging.mopad_wrapper import Beach
from obspy.signal.tf_misfit import plotTfr


def plot_events(events, map_object, beachball_size=0.02, project=None):
    """
    """
    for event in events:
        # Add beachball plot.
        x, y = map_object(event["longitude"], event["latitude"])

        focmec = [event["m_rr"], event["m_tt"], event["m_pp"], event["m_rt"],
                  event["m_rp"], event["m_tp"]]
        # Attempt to calculate the best beachball size.
        width = max((map_object.xmax - map_object.xmin,
                     map_object.ymax - map_object.ymin)) * beachball_size
        b = Beach(focmec, xy=(x, y), width=width, linewidth=1, facecolor="red")

        b.set_zorder(200000000)
        plt.gca().add_collection(b)


def plot_raydensity(map_object, station_events, domain):
    """
    Create a ray-density plot for all events and all stations.

    This function is potentially expensive and will use all CPUs available.
    Does require geographiclib to be installed.
    """
    import ctypes as C
    from lasif.tools.great_circle_binner import GreatCircleBinner
    from lasif.utils import Point
    import multiprocessing
    import progressbar
    from scipy.stats import scoreatpercentile

    bounds = domain.get_max_extent()

    # Merge everything so that a list with coordinate pairs is created. This
    # list is then distributed among all processors.
    station_event_list = []
    for event, stations in station_events:
        e_point = Point(event["latitude"], event["longitude"])
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
        new_bins = GreatCircleBinner(
            bounds["minimum_latitude"], bounds["maximum_latitude"],
            lat_lng_count, bounds["minimum_longitude"],
            bounds["maximum_longitude"], lat_lng_count)
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
    collected_bins = GreatCircleBinner(
        bounds["minimum_latitude"], bounds["maximum_latitude"], lat_lng_count,
        bounds["minimum_longitude"], bounds["maximum_longitude"],
        lat_lng_count)

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
        processes.append(multiprocessing.Process(
            target=great_circle_binning, args=(chunks[_i], collected_bins_data,
                                               collected_bins.bins.shape, lock,
                                               counter)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    pbar.finish()

    title = "%i Events with %i recorded 3 component waveforms" % (
        len(station_events), circle_count)
    # plt.gca().set_title(title, size="large")
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
    map_object.fillcontinents(color='#dddddd', lake_color='#dddddd', zorder=0)

    lngs, lats = collected_bins.coordinates
    ln, la = map_object(lngs, lats)
    map_object.pcolormesh(ln, la, data, cmap=cmap, vmin=0, vmax=max_val)
    # Draw the coastlines so they appear over the rays. Otherwise things are
    # sometimes hard to see.
    map_object.drawcoastlines()
    map_object.drawcountries(linewidth=0.2)


def plot_stations_for_event(map_object, station_dict, event_info):
    """
    Plots all stations for one event.

    :param station_dict: A dictionary whose values at least contain latitude
        and longitude keys.
    """
    import re

    # Loop as dicts are unordered.
    lngs = []
    lats = []
    station_ids = []
    for key, value in station_dict.iteritems():
        lngs.append(value["longitude"])
        lats.append(value["latitude"])
        station_ids.append(key)

    x, y = map_object(lngs, lats)

    stations = map_object.scatter(x, y, color="green", s=35, marker="v",
                                  zorder=100, edgecolor="black")
    # Setting the picker overwrites the edgecolor attribute on certain
    # matplotlib and basemap versions. Fix it here.
    stations._edgecolors = np.array([[0.0, 0.0, 0.0, 1.0]])
    stations._edgecolors_original = "black"

    # Plot the ray paths.
    for sta_lng, sta_lat in izip(lngs, lats):
        map_object.drawgreatcircle(event_info["longitude"],
                                   event_info["latitude"], sta_lng, sta_lat,
                                   lw=2, alpha=0.3)

    title = "Event in %s, at %s, %.1f Mw, with %i stations." % (
        event_info["region"], re.sub(
            r":\d{2}\.\d{6}Z", "", str(event_info["origin_time"])),
        event_info["magnitude"], len(station_dict))
    plt.gca().set_title(title, size="large")


def plot_tf(data, delta, freqmin=None, freqmax=None):
    """
    Plots a time frequency representation of any time series. Right now it is
    basically limited to plotting source time functions.
    """
    npts = len(data)

    fig = plotTfr(data, dt=delta, fmin=1.0 / (npts * delta),
                  fmax=1.0 / (2.0 * delta), show=False)

    # Get the different axes...use some kind of logic to determine which is
    # which. This is super flaky as dependent on the ObsPy version and what
    # not.
    axes = {}
    for ax in fig.axes:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Colorbar.
        if xlim == ylim:
            continue

        # Spectral axis.
        elif xlim[0] > xlim[1]:
            axes["spec"] = ax

        elif ylim[0] < 0:
            axes["time"] = ax

        else:
            axes["tf"] = ax

    fig.suptitle("Source Time Function")

    if len(axes) != 3:
        msg = "Could not plot frequency limits!"
        print msg
        plt.gcf().patch.set_alpha(0.0)
        plt.show()
        return

    axes["spec"].grid()
    axes["time"].grid()
    axes["tf"].grid()

    axes["spec"].xaxis.tick_top()
    axes["spec"].set_ylabel("Frequency [Hz]")

    axes["time"].set_xlabel("Time [s]")
    axes["time"].set_ylabel("Velocity [m/s]")

    if freqmin is not None and freqmax is not None:
        xmin, xmax = axes["tf"].get_xlim()
        axes["tf"].hlines(freqmin, xmin, xmax, color="green", lw=2)
        axes["tf"].hlines(freqmax, xmin, xmax, color="red", lw=2)
        axes["tf"].text(xmax - (0.02 * (xmax - xmin)),
                        freqmin,
                        "%.1f s" % (1.0 / freqmin),
                        color="green",
                        horizontalalignment="right", verticalalignment="top")
        axes["tf"].text(xmax - (0.02 * (xmax - xmin)),
                        freqmax,
                        "%.1f s" % (1.0 / freqmax),
                        color="red",
                        horizontalalignment="right",
                        verticalalignment="bottom")

        xmin, xmax = axes["spec"].get_xlim()
        axes["spec"].hlines(freqmin, xmin, xmax, color="green", lw=2)
        axes["spec"].hlines(freqmax, xmin, xmax, color="red", lw=2)

    plt.gcf().patch.set_alpha(0.0)
    plt.show()


def plot_event_histogram(events, plot_type):
    from matplotlib.dates import date2num, num2date
    from matplotlib import ticker

    plt.figure(figsize=(12, 4))

    values = []
    for event in events:
        if plot_type == "depth":
            values.append(event["depth_in_km"])
        elif plot_type == "time":
            values.append(date2num(event["origin_time"].datetime))

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
