#!/usr/bin/env python
# -*- coding: utf-8 -*-
import flask
from flask.ext.cache import Cache

import matplotlib.pylab as plt
from matplotlib.colors import hex2color
plt.switch_backend("agg")

import geojson
from obspy.imaging.mopad_wrapper import Beach
import io
import inspect
import os

import lasif


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
    domain = app.project.domain
    bounds = domain["bounds"]
    outer_border = lasif.rotations.get_border_latlng_list(
        bounds["minimum_latitude"],
        bounds["maximum_latitude"], bounds["minimum_longitude"],
        bounds["maximum_longitude"], 25,
        rotation_axis=domain["rotation_axis"],
        rotation_angle_in_degree=domain["rotation_angle"])

    buf = bounds["boundary_width_in_degree"]
    inner_border = lasif.rotations.get_border_latlng_list(
        bounds["minimum_latitude"] + buf,
        bounds["maximum_latitude"] - buf,
        bounds["minimum_longitude"] + buf,
        bounds["maximum_longitude"] - buf, 25,
        rotation_axis=domain["rotation_axis"],
        rotation_angle_in_degree=domain["rotation_angle"])


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
        "project_name": app.project.config["name"],
        "project_root": app.project.paths["root"]
    }
    return flask.jsonify(**info)


@app.route("/rest/latest_output")
def get_output():
    """
    Returns a list of the latest outputs.
    """
    import glob

    output_folder = app.project.paths["output"]
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

    bb = Beach(focmec, xy=(0, 0), width=200, linewidth=lw, facecolor=color)
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


@app.route("/rest/event")
def list_events():
    """
    Returns a list of events.
    """
    events = dict(app.project.events)
    for value in events.itervalues():
        value["origin_time"] = str(value["origin_time"])
    return flask.jsonify(events=events.values())


@app.route("/rest/event/<event_name>")
def get_event_details(event_name):
    stations = app.project.get_stations_for_event(event_name);
    for key, value in stations.iteritems():
        value["station_name"] = key
    return flask.jsonify(stations=stations.values())


@app.route("/")
def index():
    filename = os.path.join(WEBSERVER_DIRECTORY, "static", "index.html")
    with open(filename, "rt") as fh:
        data = fh.read()
    return data

def serve(project, port=8008, debug=False):
    cache.init_app(app, config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": os.path.join(project.paths["cache"], "webapp_cache")})

    app.project = project
    app.run(port=port, debug=debug)
