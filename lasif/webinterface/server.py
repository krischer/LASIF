#!/usr/bin/env python
# -*- coding: utf-8 -*-
import flask
from flask.ext.cache import Cache

import matplotlib.pylab as plt
from matplotlib.colors import hex2color
plt.switch_backend("agg")

import collections
import copy
import geojson
import json
from obspy.imaging.beachball import beach
import io
import inspect
import numpy as np
import os


WEBSERVER_DIRECTORY = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe())))
STATIC_DIRECTORY = os.path.join(WEBSERVER_DIRECTORY, "static")

app = flask.Flask("LASIF Webinterface", static_folder=STATIC_DIRECTORY)
cache = Cache()


def make_cache_key(*args, **kwargs):
    path = flask.request.path
    args = str(hash(frozenset(flask.request.args.items())))
    return (path + args).encode('utf-8')


@app.route("/rest/domain.geojson")
def get_domain_geojson():
    """
    Return the domain as GeoJSON multipath.
    """
    domain = app.comm.project.domain

    outer_border = domain.border
    inner_border = domain.inner_border

    border = geojson.MultiLineString([
        [(_i[1], _i[0]) for _i in inner_border],
        [(_i[1], _i[0]) for _i in outer_border],
    ])
    return flask.jsonify(**border)


@app.route("/rest/info")
def get_info():
    """
    Returns some basic information about the project.
    """
    info = {
        "project_name": app.comm.project.config["name"],
        "project_root": app.comm.project.paths["root"]
    }
    return flask.jsonify(**info)


@app.route("/rest/latest_output")
def get_output():
    """
    Returns a list of the latest outputs.
    """
    import glob

    output_folder = app.comm.project.paths["output"]
    folders = glob.glob(os.path.join(output_folder, "????-??-??*"))
    folders = sorted((os.path.basename(i) for i in folders))[-7:][::-1]
    all_folders = []
    for folder in folders:
        split = folder.split("___")
        if len(split) == 2:
            details = ""
        elif len(split) == 3:
            details = split[-1]
        else:
            flask.abort(500)
        all_folders.append({
            "time": split[0],
            "type": split[1],
            "details": details
        })

    return flask.jsonify(folders=all_folders)


@app.route("/plot/mt")
@cache.cached(key_prefix=make_cache_key)
def mt_plot():
    """
    Return a moment tensor image.
    """
    formats = {
        "png": "image/png",
        "svg": "image/svg+xml"
    }

    args = flask.request.args
    m_rr = float(args["m_rr"])
    m_tt = float(args["m_tt"])
    m_pp = float(args["m_pp"])
    m_rt = float(args["m_rt"])
    m_rp = float(args["m_rp"])
    m_tp = float(args["m_tp"])
    focmec = (m_rr, m_tt, m_pp, m_rt, m_rp, m_tp)

    # Allow hexcolors.
    color = args.get("color", "red")
    try:
        hexcolor = "#" + color
        hex2color(hexcolor)
        color = hexcolor
    except ValueError:
        pass

    size = int(args.get("size", 32))
    lw = float(args.get("lw", 1))
    format = args.get("format", "png")

    if format not in formats.keys():
        flask.abort(500)

    dpi = 100
    fig = plt.figure(figsize=(float(size) / float(dpi),
                              float(size) / float(dpi)),
                     dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    bb = beach(focmec, xy=(0, 0), width=200, linewidth=lw, facecolor=color)
    ax.add_collection(bb)
    ax.set_xlim(-105, 105)
    ax.set_ylim(-105, 105)

    temp = io.BytesIO()
    plt.savefig(temp, format=format, dpi=dpi, transparent=True)
    plt.close(fig)
    plt.close("all")
    temp.seek(0, 0)

    return flask.send_file(temp, mimetype=formats[format],
                           add_etags=False,
                           attachment_filename="mt.%s" % format)


@app.route("/rest/iteration")
def list_iterations():
    """
    Returns a list of events.
    """
    iterations = app.comm.iterations.list()
    return flask.jsonify(iterations)


@app.route("/rest/windows")
def list_windows():
    """
    Returns a JSON dictionary with the events that have windows for each
    iteration.
    """
    iterations = collections.defaultdict(list)
    events = app.comm.windows.list()
    for event_name in events:
        its = app.comm.windows.list_for_event(event_name)
        for it in its:
            iterations[it].append(event_name)
    return flask.jsonify(iterations)


@app.route("/rest/window_statistics/<iteration_name>")
def get_window_statistics_for_iteration(iteration_name):
    return flask.jsonify(
        app.comm.windows.get_window_statistics(iteration_name))


@app.route("/rest/window_plot")
def get_window_plot():
    """
    Various plots related to windows.
    """
    args = flask.request.args

    plot_type = args.get("type")

    if plot_type == "window_distance":
        iteration = args.get("iteration")
        event = args.get("event")
        app.comm.visualizations.plot_windows(event=event, iteration=iteration,
                                             distance_bins=500, show=False)

    temp = io.BytesIO()
    plt.savefig(temp, format="png", dpi=200, transparent=True)
    plt.close("all")
    temp.seek(0, 0)

    return flask.send_file(temp, mimetype="image/png",
                           add_etags=False)


@app.route("/rest/iteration/<iteration_name>")
def get_iteration_detail(iteration_name):
    """
    Returns a list of events.
    """
    iteration = app.comm.iterations.get(iteration_name)

    stf = iteration.get_source_time_function()
    stf["data"] = stf["data"].tolist()

    return flask.jsonify(
        iteration_name=iteration.iteration_name,
        description=iteration.description,
        comments=iteration.comments,
        data_preprocessing=iteration.data_preprocessing,
        events=iteration.events.keys(),
        processing_params=iteration.get_process_params(),
        processing_tag=iteration.processing_tag,
        solver=iteration.solver_settings["solver"],
        solver_settings=iteration.solver_settings["solver_settings"],
        source_time_function=stf)


@app.route("/rest/iteration/<iteration_name>/stf")
def get_iteration_stf(iteration_name):
    pass


@app.route("/rest/event")
def list_events():
    """
    Returns a list of events.
    """
    events = copy.deepcopy(app.comm.events.get_all_events())
    for value in events.itervalues():
        value["origin_time"] = str(value["origin_time"])
    return flask.jsonify(events=events.values())


@app.route("/rest/event/<event_name>")
def get_event_details(event_name):
    event = copy.deepcopy(app.comm.events.get(event_name))
    event["origin_time"] = str(event["origin_time"])
    stations = app.comm.query.get_all_stations_for_event(event_name)
    for key, value in stations.iteritems():
        value["station_name"] = key
    event["stations"] = stations.values()
    return flask.jsonify(**event)


@app.route("/rest/available_data/<event_name>/<station_id>")
def get_available_data(event_name, station_id):
    available_data = app.comm.query.discover_available_data(event_name,
                                                            station_id)
    return flask.jsonify(**available_data)


@app.route("/rest/get_data/<event_name>/<station_id>/<name>")
def get_data(event_name, station_id, name):
    if name == "raw":
        st = app.comm.waveforms.get_waveforms_raw(event_name, station_id)
    elif name.startswith("preprocessed_"):
        st = app.comm.waveforms.get_waveforms_processed(
            event_name, station_id, tag=name)
    else:
        st = app.comm.waveforms.get_waveforms_synthetic(
            event_name, station_id, long_iteration_name=name)

    BIN_LENGTH = 2000
    data = {}
    components = {}

    for tr in st:
        component = tr.stats.channel[-1].upper()
        # Normalize data.
        tr.data = np.require(tr.data, dtype="float32")
        tr.data -= tr.data.min()
        tr.data /= tr.data.max()
        tr.data -= tr.data.mean()
        tr.data /= np.abs(tr.data).max() * 1.1

        if tr.stats.npts > 5000:
            rest = tr.stats.npts % BIN_LENGTH
            per_bin = tr.stats.npts // BIN_LENGTH
            if rest:
                final_data = np.empty(2 * BIN_LENGTH + 2)
                final_data[::2][:BIN_LENGTH] = tr.data[:-rest].reshape(
                    (BIN_LENGTH, per_bin)).min(axis=1)
                final_data[1::2][:BIN_LENGTH] = tr.data[:-rest].reshape(
                    (BIN_LENGTH, per_bin)).max(axis=1)
            else:
                final_data = np.empty(2 * BIN_LENGTH)
                final_data[::2][:BIN_LENGTH] = tr.data.reshape(
                    (BIN_LENGTH, per_bin)).min(axis=1)
                final_data[1::2][:BIN_LENGTH] = tr.data.reshape(
                    (BIN_LENGTH, per_bin)).max(axis=1)
            if rest:
                final_data[-2] = tr.data[-rest:].min()
                final_data[-1] = tr.data[-rest:].max()
            time_array = np.empty(len(final_data))
            time_array[::2] = np.linspace(tr.stats.starttime.timestamp,
                                          tr.stats.endtime.timestamp,
                                          len(final_data) / 2)
            time_array[1::2] = np.linspace(tr.stats.starttime.timestamp,
                                           tr.stats.endtime.timestamp,
                                           len(final_data) / 2)
        else:
            final_data = tr.data
            # Create times array.
            time_array = np.linspace(tr.stats.starttime.timestamp,
                                     tr.stats.endtime.timestamp,
                                     tr.stats.npts)

        temp = np.empty((len(final_data), 2), dtype="float64")
        temp[:, 0] = time_array
        temp[:, 1] = final_data
        components[component] = temp.tolist()

    # Much faster then flask.jsonify as it does not pretty print.
    data = json.dumps(components)
    return data


@app.route("/")
def index():
    filename = os.path.join(WEBSERVER_DIRECTORY, "static", "index.html")
    with open(filename, "rt") as fh:
        data = fh.read()
    return data


def serve(comm, port=8008, debug=False, open_to_outside=False):
    """
    Start the server.

    :param comm: LASIF communicator instance.
    :param port: The port to launch on.
    :param debug: Debug on/off.
    :param open_to_outside: By default it only serves on localhost thus the
        server cannot be accessed from other PCs. Set this to True to enable
        access from other computers.
    """
    cache.init_app(app, config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": os.path.join(comm.project.paths["cache"],
                                  "webapp_cache")})

    if open_to_outside is True:
        host = "0.0.0.0"
    else:
        host = None

    app.comm = comm
    app.run(port=port, debug=debug, host=host)
