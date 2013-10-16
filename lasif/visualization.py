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
                rotation_angle_in_degree=0.0, plot_simulation_domain=False,
                zoom=False, resolution=None, ax=None):
    """
    """
    bounds = rotations.get_max_extention_of_domain(
        min_latitude, max_latitude, min_longitude, max_longitude,
        rotation_axis=rotation_axis,
        rotation_angle_in_degree=rotation_angle_in_degree)
    center_lat = bounds["minimum_latitude"] + (
        bounds["maximum_latitude"] - bounds["minimum_latitude"]) / 2.0
    center_lng = bounds["minimum_longitude"] + (
        bounds["maximum_longitude"] - bounds["minimum_longitude"]) / 2.0

    extend_x = bounds["maximum_longitude"] - bounds["minimum_longitude"]
    extend_y = bounds["maximum_latitude"] - bounds["minimum_latitude"]
    max_extend = max(extend_x, extend_y)

    # Arbitrary threshold
    if zoom is False or max_extend > 70:
        if resolution is None:
            resolution = "c"
        m = Basemap(projection='ortho', lon_0=center_lng, lat_0=center_lat,
                    resolution=resolution, ax=ax)
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

    border = rotations.get_border_latlng_list(
        min_latitude, max_latitude, min_longitude, max_longitude,
        rotation_axis=rotation_axis,
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
        border = rotations.get_border_latlng_list(
            min_latitude, max_latitude, min_longitude, max_longitude)
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

    return m


# File global pick state handler. Used to handle the different pick events.
__pick_state = {
    # Any possible existing event_annotations.
    "event_annotations": [],
    # The currently annotated event if any.
    "current_event_annotation_artist": None
}


def _pick_handler(event):
    """
    File global pick handler. Called for every pick event in the file.
    """
    if hasattr(event.artist, "detailed_event_description"):
        __pick_handler_event_annotation(event)
    elif hasattr(event.artist, "_station_scatter"):
        __pick_handler_station_scatter(event)
    else:
        return
    plt.gcf().canvas.draw()


def __pick_handler_event_annotation(event):
    """
    Pick handler for the event annotation.
    """
    # Remove any potentially existing annotations.
    for i in __pick_state["event_annotations"]:
        i.remove()
    __pick_state["event_annotations"][:] = []

    if __pick_state["current_event_annotation_artist"] is event.artist:
        __pick_state["current_event_annotation_artist"] = None
        return

    x, y = event.mouseevent.xdata, event.mouseevent.ydata
    annotation = plt.annotate(
        event.artist.detailed_event_description,
        xy=(x, y), xytext=(0.98, 0.98), textcoords="figure fraction",
        horizontalalignment="right", verticalalignment="top",
        arrowprops=dict(arrowstyle="fancy", color="0.5",
                        connectionstyle="arc3,rad=0.3"),
        zorder=10E9, fontsize="small")
    __pick_state["event_annotations"].append(annotation)
    __pick_state["current_event_annotation_artist"] = event.artist


def __pick_handler_station_scatter(event):
    idx = event.ind[0]
    station_name = event.artist._station_scatter[idx]
    if event.artist._project:
        event.artist._project.plot_station(
            station_name, event.artist._event_info["event_name"])


def _set_global_pick_handler():
    """
    Sets the global pick handler if not yet set.
    """
    canvas = plt.gcf().canvas
    if "pick_event" in canvas.callbacks.callbacks:
        return
    canvas.mpl_connect("pick_event", _pick_handler)


def plot_events(events, map_object, beachball_size=0.02):
    """
    """
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
                     map_object.ymax - map_object.ymin)) * beachball_size
        b = Beach(focmec, xy=(x, y), width=width, linewidth=1, facecolor="red")
        b.set_picker(True)
        b.detailed_event_description = (
            "Event %.1f %s\n"
            "Lat: %.1f, Lng: %.1f, Depth: %.1f km\n"
            "Time: %s\n"
            "%s"
        ) % (mag.mag, mag.magnitude_type, org.latitude, org.longitude,
             org.depth / 1000.0, org.time, os.path.basename(event.filename))

        b.set_zorder(200000000)
        plt.gca().add_collection(b)
    _set_global_pick_handler()


def plot_raydensity(map_object, station_events, min_lat, max_lat, min_lng,
                    max_lng, rot_axis, rot_angle):
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

    bounds = rotations.get_max_extention_of_domain(
        min_lat, max_lat, min_lng, max_lng, rotation_axis=rot_axis,
        rotation_angle_in_degree=rot_angle)

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


def plot_data_for_station(station, raw_files, processed_files,
                          synthetic_files, event, project=None):
    """
    """
    import datetime
    import matplotlib.dates
    from matplotlib.widgets import CheckButtons
    from obspy import read
    from obspy.core.util.geodetics import calcVincentyInverse
    import textwrap

    fig = plt.figure(figsize=(14, 9))
    fig.canvas.set_window_title("Data for station %s" % station["id"])

    fig.text(0.5, 0.99, "Station %s" % station["id"],
             verticalalignment="top", horizontalalignment="center")

    # Add one axis for each component. Share all axes.
    z_axis = fig.add_axes([0.30, 0.65, 0.68, 0.3])
    n_axis = fig.add_axes([0.30, 0.35, 0.68, 0.3], sharex=z_axis,
                          sharey=z_axis)
    e_axis = fig.add_axes([0.30, 0.05, 0.68, 0.3], sharex=z_axis,
                          sharey=z_axis)
    axis = [z_axis, n_axis, e_axis]

    # Set grid, autoscale and hide all tick labels (some will be made visible
    # later one)
    for axes in axis:
        plt.setp(axes.get_xticklabels(), visible=False)
        plt.setp(axes.get_yticklabels(), visible=False)
        axes.grid(b=True)
        axes.autoscale(enable=True)
        axes.set_xlim(0.0, 12345.0)

    # Axes for the data selection check boxes.
    raw_check_axes = fig.add_axes([0.01, 0.8, 0.135, 0.15])
    synth_check_axes = fig.add_axes([0.155, 0.8, 0.135, 0.15])
    proc_check_axes = fig.add_axes([0.01, 0.5, 0.28, 0.29])

    # The map axes
    map_axes = fig.add_axes([0.01, 0.05, 0.28, 0.40])

    # Fill the check box axes.
    raw_check = CheckButtons(raw_check_axes, ["raw"], [True])
    proc_check = CheckButtons(proc_check_axes, [
        "\n".join(textwrap.wrap(_i, width=30))
        for _i in processed_files.keys()],
        [False] * len(processed_files))
    synth_check = CheckButtons(synth_check_axes, synthetic_files.keys(),
                               [False] * len(synthetic_files))

    for check in [raw_check, proc_check, synth_check]:
        plt.setp(check.labels, fontsize=10)

    raw_check_axes.text(
        0.02, 0.97, "Raw Data", transform=raw_check_axes.transAxes,
        verticalalignment="top", horizontalalignment="left", fontsize=10)
    proc_check_axes.text(
        0.02, 0.97, "Processed Data", transform=proc_check_axes.transAxes,
        verticalalignment="top", horizontalalignment="left", fontsize=10)
    synth_check_axes.text(
        0.02, 0.97, "Synthetic Data", transform=synth_check_axes.transAxes,
        verticalalignment="top", horizontalalignment="left", fontsize=10)

    if project:
        bounds = project.domain["bounds"]
        map_object = plot_domain(
            bounds["minimum_latitude"], bounds["maximum_latitude"],
            bounds["minimum_longitude"], bounds["maximum_longitude"],
            bounds["boundary_width_in_degree"],
            rotation_axis=project.domain["rotation_axis"],
            rotation_angle_in_degree=project.domain["rotation_angle"],
            plot_simulation_domain=False, zoom=True, ax=map_axes)

        plot_stations_for_event(map_object=map_object,
                                station_dict={station["id"]: station},
                                event_info=event)
        # Plot the beachball for one event.
        plot_events([project.get_event(event["event_name"])],
                    map_object=map_object, beachball_size=0.05)
        dist = calcVincentyInverse(
            event["latitude"], event["longitude"], station["latitude"],
            station["longitude"])[0] / 1000.0

        map_axes.set_title("Epicentral distance: %.1f km | Mag: %.1f %s" %
                           (dist, event["magnitude"], event["magnitude_type"]),
                           fontsize=10)

    SYNTH_MAPPING = {"X": "N", "Y": "E", "Z": "Z"}

    PLOT_OBJECTS = {
        "raw": None,
        "synthetics": {},
        "processed": {}
    }

    def plot(plot_type, filename, save_at):
        tr = read(filename)[0]
        tr.data = np.require(tr.data, dtype="float32")
        tr.data -= tr.data.min()
        tr.data /= tr.data.max()
        tr.data -= tr.data.mean()
        tr.data /= np.abs(tr.data).max() * 1.1

        component = tr.stats.channel[-1].upper()
        if component in SYNTH_MAPPING:
            component = SYNTH_MAPPING[component]

        if plot_type == "synthetic" and component in ["X", "Z"]:
            tr.data *= -1

        if component == "N":
            axis = n_axis
        elif component == "E":
            axis = e_axis
        elif component == "Z":
            axis = z_axis
        else:
            raise NotImplementedError

        if plot_type == "synthetic":
            time_axis = matplotlib.dates.drange(
                event["origin_time"].datetime,
                (event["origin_time"] + tr.stats.delta * (tr.stats.npts))
                .datetime,
                datetime.timedelta(seconds=tr.stats.delta))
            zorder = 2
            color = "red"
        elif plot_type == "raw":
            time_axis = matplotlib.dates.drange(
                tr.stats.starttime.datetime,
                (tr.stats.endtime + tr.stats.delta).datetime,
                datetime.timedelta(seconds=tr.stats.delta))
            zorder = 0
            color = "0.8"
        elif plot_type == "processed":
            time_axis = matplotlib.dates.drange(
                tr.stats.starttime.datetime,
                (tr.stats.endtime + tr.stats.delta).datetime,
                datetime.timedelta(seconds=tr.stats.delta))
            zorder = 1
            color = "0.2"
        else:
            msg = "Plot type '%s' not known" % plot_type
            raise ValueError(msg)

        save_at.append(axis.plot_date(time_axis[:len(tr.data)], tr.data,
                       color=color, zorder=zorder, marker="", linestyle="-"))
        axis.set_ylim(-1.0, 1.0)
        axis.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%H:%M:%S"))

        if component == "E":
            try:
                plt.setp(axis.get_xticklabels(), visible=True)
            except:
                pass

        if plot_type != "raw":
            axis.set_xlim(time_axis[0], time_axis[-1])
        # Adjust the limit only if there are no synthetics and processed if
        # plotting raw data.
        elif not PLOT_OBJECTS["synthetics"] and not PLOT_OBJECTS["processed"]:
            axis.set_xlim(time_axis[0], time_axis[-1])

    for label, axis in zip(("Vertical", "North", "East"),
                           (z_axis, n_axis, e_axis)):
        axis.text(0.98, 0.95, label,
                  verticalalignment="top", horizontalalignment="right",
                  bbox=dict(facecolor="white", alpha=0.5, pad=5),
                  transform=axis.transAxes, fontsize=11)

    def _checked_raw(label):
        checked(label, "raw")

    def _checked_proc(label):
        checked(label, "proc")

    def _checked_synth(label):
        checked(label, "synth")

    def checked(label, check_box):
        if check_box == "raw":
            if PLOT_OBJECTS["raw"] is not None:
                for _i in PLOT_OBJECTS["raw"]:
                    for _j in _i:
                        _j.remove()
                PLOT_OBJECTS["raw"] = None
            else:
                PLOT_OBJECTS["raw"] = []
                for filename in raw_files:
                    plot("raw", filename, PLOT_OBJECTS["raw"])
        elif check_box == "synth":
            if label in PLOT_OBJECTS["synthetics"]:
                for _i in PLOT_OBJECTS["synthetics"][label]:
                    for _j in _i:
                        _j.remove()
                del PLOT_OBJECTS["synthetics"][label]
            else:
                PLOT_OBJECTS["synthetics"][label] = []
                for filename in synthetic_files[label]:
                    plot("synthetic", filename,
                         PLOT_OBJECTS["synthetics"][label])
        elif check_box == "proc":
            # Previously broken up.
            label = label.replace("\n", "")
            if label in PLOT_OBJECTS["processed"]:
                for _i in PLOT_OBJECTS["processed"][label]:
                    for _j in _i:
                        _j.remove()
                del PLOT_OBJECTS["processed"][label]
            else:
                PLOT_OBJECTS["processed"][label] = []
                for filename in processed_files[label]:
                    plot("processed", filename,
                         PLOT_OBJECTS["processed"][label])

        try:
            fig.canvas.draw()
        except:
            pass

    raw_check.on_clicked(_checked_raw)
    proc_check.on_clicked(_checked_proc)
    synth_check.on_clicked(_checked_synth)

    # Always plot the raw data by default.
    _checked_raw("raw")

    try:
        fig.canvas.draw()
    except:
        pass

    # One has call plt.show() to activate the main loop of the new figure.
    # Otherwise events will not work.
    plt.show()


def plot_stations_for_event(map_object, station_dict, event_info,
                            project=None):
    """
    Plots all stations for one event.

    :param station_dict: A dictionary whose values at least contain latitude
        and longitude keys.
    """
    # Loop as dicts are unordered.
    lngs = []
    lats = []
    station_names = []
    for key, value in station_dict.iteritems():
        lngs.append(value["longitude"])
        lats.append(value["latitude"])
        station_names.append(key)

    x, y = map_object(lngs, lats)

    stations = map_object.scatter(x, y, color="green", s=35, marker="v",
                                  zorder=100, edgecolor="black", picker=1.0)
    # Setting the picker overwrites the edgecolor attribute on certain
    # matplotlib and basemap versions. Fix it here.
    stations._edgecolors = np.array([[0.0, 0.0, 0.0, 1.0]])
    stations._edgecolors_original = "black"
    # Add three additional information attributes that the pick callback has
    # access to.
    stations._station_scatter = station_names
    stations._project = project
    stations._event_info = event_info

    # Plot the ray paths.
    for sta_lng, sta_lat in izip(lngs, lats):
        map_object.drawgreatcircle(event_info["longitude"],
                                   event_info["latitude"], sta_lng, sta_lat,
                                   lw=2, alpha=0.3)

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
