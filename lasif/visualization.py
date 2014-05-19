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

    # If the simulation domain is also available, use it to calculate the
    # max_extend. This results in the simulation domain affecting the zoom
    # level.
    if plot_simulation_domain is True:
        simulation_domain = rotations.get_border_latlng_list(
            min_latitude, max_latitude, min_longitude, max_longitude)
        simulation_domain = np.array(simulation_domain)
        min_y, min_x = simulation_domain.min(axis=0)
        max_y, max_x = simulation_domain.max(axis=0)
        max_extend = max(
            max(max_x, bounds["maximum_longitude"]) -
            min(min_x, bounds["minimum_longitude"]),
            max(max_y, bounds["maximum_latitude"]) -
            min(min_y, bounds["minimum_latitude"])
        )

    # Arbitrary threshold
    if zoom is False or max_extend > 90:
        if resolution is None:
            resolution = "c"
        m = Basemap(projection='ortho', lon_0=center_lng, lat_0=center_lat,
                    resolution=resolution, ax=ax)

        parallels = np.arange(-90.0, 90.0, 10.0)
        m.drawparallels(parallels)

        meridians = np.arange(0.0, 351.0, 10.0)
        m.drawmeridians(meridians)

    else:
        if resolution is None:
            resolution = "l"
        # Calculate approximate width and height in meters.
        width = bounds["maximum_longitude"] - bounds["minimum_longitude"]
        height = bounds["maximum_latitude"] - bounds["minimum_latitude"]

        if width > 50.0:
            factor = 10.0
        elif (width <= 50.0) & (width > 20.0):
            factor = 5.0
        elif (width <= 20.0) & (width > 5.0):
            factor = 2.0
        else:
            factor = 1.0

        meridians = np.arange(
            factor * np.round(bounds["minimum_longitude"] / factor) - factor,
            factor * np.round(bounds["maximum_longitude"] / factor + factor),
            factor)
        parallels = np.arange(
            factor * np.round(bounds["minimum_latitude"] / factor) - factor,
            factor * np.round(bounds["maximum_latitude"] / factor + factor),
            factor)

        width *= 110000 * 1.1
        height *= 110000 * 1.3
        # Lambert azimuthal equal area projection. Equal area projections
        # are useful for interpreting features and this particular one also
        # does not distort features a lot on regional scales.
        m = Basemap(projection='laea', resolution=resolution, width=width,
                    height=height, lat_0=center_lat, lon_0=center_lng)
        m.drawparallels(parallels, labels=[False, True, False, False])
        m.drawmeridians(meridians, labels=[False, False, False, True])

    m.drawmapboundary(fill_color='#cccccc')
    m.fillcontinents(color='white', lake_color='#cccccc', zorder=0)
    # m.drawcoastlines()

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
        lats = simulation_domain[:, 0]
        lngs = simulation_domain[:, 1]
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

    plt.gcf().patch.set_alpha(0.0)
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
    if event.mouseevent.button == 1 and not event.mouseevent.dblclick:
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
    # If it is a double-click, plot the event in a new figure.
    elif event.mouseevent.dblclick and event.artist._project:
        plt.figure()
        event.artist._project.plot_event(event.artist._event_name)
        plt.gcf().patch.set_alpha(0.0)
        plt.show()


def __pick_handler_station_scatter(event):
    idx = event.ind[0]
    station_id = event.artist._station_scatter[idx]
    if event.artist._project:
        event.artist._project.plot_station(
            station_id, event.artist._event_info["event_name"])


def _set_global_pick_handler():
    """
    Sets the global pick handler if not yet set.
    """
    canvas = plt.gcf().canvas
    if "pick_event" in canvas.callbacks.callbacks:
        return
    canvas.mpl_connect("pick_event", _pick_handler)


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
        b.set_picker(True)
        b._project = project
        b._event_name = os.path.splitext(
            os.path.basename(event["filename"]))[0]
        b.detailed_event_description = (
            "Event %.1f %s\n"
            "Lat: %.1f, Lng: %.1f, Depth: %.1f km\n"
            "Time: %s\n"
            "%s"
        ) % (event["magnitude"], event["magnitude_type"], event["latitude"],
             event["longitude"], event["depth_in_km"], event["origin_time"],
             event["event_name"])

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


def plot_data_for_station(station, available_data, event, get_data_callback,
                          domain_bounds):
    """
    Plots all data for a station in an interactive plot.

    :type station: dict
    :param station: A dictionary containing the keys 'id', 'latitude',
        'longitude', 'elevation_in_m', and 'local_depth_in_m' describing the
        current station.
    :type available_data: dict
    :param available_data: The available processed and synthetic data. The raw
        data is always assumed to be available.
    :type event: dict
    :param event: A dictionary describing the current event.
    :type get_data_callback: function
    :param get_data_callback: Callback function returning an ObsPy Stream
        object.

        get_data_callback("raw")
        get_data_callback("synthetic", iteration_name)
        get_data_callback("processed", processing_tag)

    :type domain_bounds: dict
    :param domain_bounds: The domain bounds.
    """
    import datetime
    import matplotlib.dates
    from matplotlib.widgets import CheckButtons
    from obspy.core.util.geodetics import calcVincentyInverse
    import textwrap

    # Setup the figure, the window and plot title.
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
        for _i in available_data["processed"]],
        [False] * len(available_data["processed"]))
    synth_check = CheckButtons(synth_check_axes, available_data["synthetic"],
                               [False] * len(available_data["synthetic"]))

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

    bounds = domain_bounds["bounds"]
    map_object = plot_domain(
        bounds["minimum_latitude"], bounds["maximum_latitude"],
        bounds["minimum_longitude"], bounds["maximum_longitude"],
        bounds["boundary_width_in_degree"],
        rotation_axis=domain_bounds["rotation_axis"],
        rotation_angle_in_degree=domain_bounds["rotation_angle"],
        plot_simulation_domain=False, zoom=True, ax=map_axes)

    plot_stations_for_event(map_object=map_object,
                            station_dict={station["id"]: station},
                            event_info=event)
    # Plot the beachball for one event.
    plot_events([event], map_object=map_object, beachball_size=0.05)
    dist = calcVincentyInverse(
        event["latitude"], event["longitude"], station["latitude"],
        station["longitude"])[0] / 1000.0

    map_axes.set_title("Epicentral distance: %.1f km | Mag: %.1f %s" %
                       (dist, event["magnitude"], event["magnitude_type"]),
                       fontsize=10)

    PLOT_OBJECTS = {
        "raw": None,
        "synthetics": {},
        "processed": {}
    }

    def plot(plot_type, label=None):
        if plot_type == "raw":
            st = get_data_callback("raw")
            PLOT_OBJECTS["raw"] = []
            save_at = PLOT_OBJECTS["raw"]
        elif plot_type == "synthetic":
            st = get_data_callback("synthetic", label)
            PLOT_OBJECTS["synthetics"][label] = []
            save_at = PLOT_OBJECTS["synthetics"][label]
        elif plot_type == "processed":
            st = get_data_callback("processed", label)
            PLOT_OBJECTS["processed"][label] = []
            save_at = PLOT_OBJECTS["processed"][label]

        # Loop over all traces.
        for tr in st:
            # Normalize data.
            tr.data = np.require(tr.data, dtype="float32")
            tr.data -= tr.data.min()
            tr.data /= tr.data.max()
            tr.data -= tr.data.mean()
            tr.data /= np.abs(tr.data).max() * 1.1

            # Figure out the correct axis.
            component = tr.stats.channel[-1].upper()
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
                           color=color, zorder=zorder, marker="",
                           linestyle="-"))
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
            elif not PLOT_OBJECTS["synthetics"] and \
                    not PLOT_OBJECTS["processed"]:
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
                plot("raw")
        elif check_box == "synth":
            if label in PLOT_OBJECTS["synthetics"]:
                for _i in PLOT_OBJECTS["synthetics"][label]:
                    for _j in _i:
                        _j.remove()
                del PLOT_OBJECTS["synthetics"][label]
            else:
                PLOT_OBJECTS["synthetics"][label] = []
                plot("synthetic", label=label)
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
                plot("processed", label=label)

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
    plt.gcf().patch.set_alpha(0.0)
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
    station_ids = []
    for key, value in station_dict.iteritems():
        lngs.append(value["longitude"])
        lats.append(value["latitude"])
        station_ids.append(key)

    x, y = map_object(lngs, lats)

    stations = map_object.scatter(x, y, color="green", s=35, marker="v",
                                  zorder=100, edgecolor="black", picker=1.0)
    # Setting the picker overwrites the edgecolor attribute on certain
    # matplotlib and basemap versions. Fix it here.
    stations._edgecolors = np.array([[0.0, 0.0, 0.0, 1.0]])
    stations._edgecolors_original = "black"
    # Add three additional information attributes that the pick callback has
    # access to.
    stations._station_scatter = station_ids
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


def plot_tf(data, delta, freqmin=None, freqmax=None):
    """
    Plots a time frequency representation of any time series. Right now it is
    basically limited to plotting source time functions.
    """
    import collections

    npts = len(data)

    fig = plotTfr(data, dt=delta, fmin=1.0 / (npts * delta),
                  fmax=1.0 / (2.0 * delta), show=False)

    x_axis = collections.defaultdict(list)
    y_axis = collections.defaultdict(list)
    for ax in fig.axes:
        x_axis[ax.get_xlim()].append(ax)
        y_axis[ax.get_ylim()].append(ax)

    # Get those, that have two items.
    x_axis = {key: value for key, value in x_axis.iteritems()
              if len(value) == 2}
    y_axis = {key: value for key, value in y_axis.iteritems()
              if len(value) == 2}

    fig.suptitle("Source Time Function")

    if len(x_axis) != 1 or len(y_axis) != 1:
        msg = "Could not plot frequency limits!"
        print msg
        plt.gcf().patch.set_alpha(0.0)
        plt.show()
        return

    # The axis that is in both collections will be the time frequency axis. The
    # other one that shares the y axis the spectrum.
    tf_axis = set(x_axis.values()[0]).intersection(
        set(y_axis.values()[0])).pop()
    spec_axis = [i for i in y_axis.values()[0] if i != tf_axis][0]
    time_axis = [i for i in x_axis.values()[0] if i != tf_axis][0]

    spec_axis.grid()
    time_axis.grid()
    tf_axis.grid()

    spec_axis.xaxis.tick_top()
    spec_axis.set_ylabel("Frequency [Hz]")

    time_axis.set_xlabel("Time [s]")
    time_axis.set_ylabel("Velocity [m/s]")

    if freqmin is not None and freqmax is not None:
        xmin, xmax = tf_axis.get_xlim()
        tf_axis.hlines(freqmin, xmin, xmax, color="green", lw=2)
        tf_axis.hlines(freqmax, xmin, xmax, color="red", lw=2)

        tf_axis.text(xmax - (0.02 * (xmax - xmin)),
                     freqmin,
                     "%.1f s" % (1.0 / freqmin),
                     color="green",
                     horizontalalignment="right", verticalalignment="top")
        tf_axis.text(xmax - (0.02 * (xmax - xmin)),
                     freqmax,
                     "%.1f s" % (1.0 / freqmax),
                     color="red",
                     horizontalalignment="right", verticalalignment="bottom")

        xmin, xmax = spec_axis.get_xlim()
        spec_axis.hlines(freqmin, xmin, xmax, color="green", lw=2)
        spec_axis.hlines(freqmax, xmin, xmax, color="red", lw=2)

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
