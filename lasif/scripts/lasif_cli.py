#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The main LASIF console script.

It is important to import necessary things at the method level to make
importing this file as fast as possible. Otherwise using the command line
interface feels sluggish and slow.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import os
import random
import shutil
import sys
import traceback

import colorama

from lasif.project import Project


FCT_PREFIX = "lasif_"


class LASIFCommandLineException(Exception):
    pass


def _find_project_root(folder):
    """
    Will search upwards from the given folder until a folder containing a
    LASIF root structure is found. The absolute path to the root is returned.
    """
    max_folder_depth = 10
    folder = folder
    for _ in xrange(max_folder_depth):
        if os.path.exists(os.path.join(folder, "config.xml")):
            return Project(os.path.abspath(folder))
        folder = os.path.join(folder, os.path.pardir)
    msg = "Not inside a LASIF project."
    raise LASIFCommandLineException(msg)


def lasif_plot_domain(args):
    """
    Usage: lasif plot_domain

    Plots the project's domain on a map.
    """
    proj = _find_project_root(".")
    proj.plot_domain()


def lasif_plot_event(args):
    """
    Usage: lasif plot_event EVENT_NAME

    Plots one event and raypaths on a map.
    """
    proj = _find_project_root(".")

    if len(args) != 1:
        msg = "EVENT_NAME must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    event_name = args[0]

    proj.plot_event(event_name)


def lasif_plot_events(args):
    """
    Usage: lasif plot_events

    Plots all events.
    """
    proj = _find_project_root(".")
    proj.plot_events()


def lasif_plot_raydensity(args):
    """
    Usage: lasif plot_raydensity

    Plots a binned raycoverage plot for all events.
    """
    proj = _find_project_root(".")
    proj.plot_raydensity()


def lasif_add_spud_event(args):
    """
    Usage: lasif add_spud_event URL

    Adds an event from the IRIS SPUD GCMT webservice to the project. URL is any
    SPUD momenttensor URL.
    """
    from lasif.scripts.iris2quakeml import iris2quakeml

    proj = _find_project_root(".")
    if len(args) != 1:
        msg = "URL must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    url = args[0]
    iris2quakeml(url, proj.paths["events"])


def lasif_info(args):
    """
    Usage: lasif info

    Print information about the current project.
    """
    proj = _find_project_root(".")
    print(proj)


def lasif_download_waveforms(args):
    """
    Usage: lasif download_waveforms EVENT_NAME

    Attempts to download all missing waveform files for a given event. The list
    of possible events can be obtained with "lasif list_events". The files will
    be saved in the DATA/EVENT_NAME/raw directory.
    """
    from lasif.download_helpers import downloader

    proj = _find_project_root(".")
    events = proj.get_event_dict()
    if len(args) != 1:
        msg = "EVENT_NAME must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    event_name = args[0]
    if event_name not in events:
        msg = "Event '%s' not found." % event_name
        raise LASIFCommandLineException(msg)

    event = proj.get_event(event_name)
    origin = event.preferred_origin() or event.origins[0]
    time = origin.time
    starttime = time - proj.config["download_settings"]["seconds_before_event"]
    endtime = time + proj.config["download_settings"]["seconds_after_event"]

    domain = proj.domain
    min_lat, max_lat, min_lng, max_lng, buffer = (
        domain["bounds"]["minimum_latitude"],
        domain["bounds"]["maximum_latitude"],
        domain["bounds"]["minimum_longitude"],
        domain["bounds"]["maximum_longitude"],
        domain["bounds"]["boundary_width_in_degree"])
    min_lat += buffer
    max_lat -= buffer
    min_lng += buffer
    max_lng -= buffer

    download_folder = os.path.join(proj.paths["data"], event_name, "raw")
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    channel_priority_list = ["HH[Z,N,E]", "BH[Z,N,E]", "MH[Z,N,E]",
        "EH[Z,N,E]", "LH[Z,N,E]"]

    logfile = os.path.join(proj.paths["logs"], "waveform_download_log.txt")

    downloader.download_waveforms(min_lat, max_lat, min_lng, max_lng,
        domain["rotation_axis"], domain["rotation_angle"], starttime, endtime,
        proj.config["download_settings"]["arclink_username"],
        channel_priority_list=channel_priority_list, logfile=logfile,
        download_folder=download_folder, waveform_format="mseed")


def lasif_download_stations(args):
    """
    Usage: lasif download_stations EVENT_NAME

    Attempts to download all missing station data files for a given event. The
    list of possible events can be obtained with "lasif list_events". The files
    will be saved in the STATION/*.
    """
    from lasif.download_helpers import downloader
    proj = _find_project_root(".")
    events = proj.get_event_dict()
    if len(args) != 1:
        msg = "EVENT_NAME must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    event_name = args[0]
    if event_name not in events:
        msg = "Event '%s' not found." % event_name
        raise LASIFCommandLineException(msg)

    # Fix the start- and endtime to ease download file grouping
    event_info = proj.get_event_info(event_name)
    starttime = event_info["origin_time"] - \
        proj.config["download_settings"]["seconds_before_event"]
    endtime = event_info["origin_time"] + \
        proj.config["download_settings"]["seconds_after_event"]
    time = starttime + (endtime - starttime) * 0.5

    # Get all channels.
    channels = proj._get_waveform_cache_file(event_name, "raw").get_values()
    channels = [{"channel_id": _i["channel_id"], "network": _i["network"],
        "station": _i["station"], "location": _i["location"],
        "channel": _i["channel"], "starttime": starttime, "endtime": endtime}
        for _i in channels]
    channels_to_download = []

    # Filter for channel not actually available
    for channel in channels:
        if proj.has_station_file(channel["channel_id"], time):
            continue
        channels_to_download.append(channel)

    downloader.download_stations(channels_to_download, proj.paths["resp"],
        proj.paths["station_xml"], proj.paths["dataless_seed"],
        logfile=os.path.join(proj.paths["logs"], "station_download_log.txt"),
        arclink_user=proj.config["download_settings"]["arclink_username"],
        get_station_filename_fct=proj.get_station_filename)


def lasif_list_events(args):
    """
    usage: lasif list_events

    returns a list of all events in the project.
    """
    events = _find_project_root(".").get_event_dict()
    print("%i event%s in project:" % (len(events), "s" if len(events) > 1
        else ""))
    for event in events.iterkeys():
        print ("\t%s" % event)


def lasif_list_models(args):
    """
    Usage: lasif list_models

    Returns a list of all models in the project.
    """
    models = _find_project_root(".").get_model_dict()
    print("%i model%s in project:" % (len(models), "s" if len(models) > 1
        else ""))
    for model in models.iterkeys():
        print ("\t%s" % model)


def lasif_plot_kernel(args):
    """
    Usage lasif plot_kernel ITERATION_NAME EVENT_NAME
    """
    from glob import glob
    from lasif import ses3d_models

    if len(args) != 2:
        msg = "ITERATION_NAME and EVENT_NAME must be given."
        raise LASIFCommandLineException(msg)

    iteration_name = args[0]
    event_name = args[1]

    proj = _find_project_root(".")

    kernel_dir = proj.get_kernel_dir(iteration_name, event_name)

    # Check if the kernel directory contains a boxfile,
    # if not search all model directories for one that fits. If none is
    # found, raise an error.
    model_directories = proj.get_model_dict().values()
    boxfile = os.path.join(kernel_dir, "boxfile")
    if not os.path.exists(boxfile):
        boxfile_found = False
        # Find all vp gradients.
        vp_gradients = glob(os.path.join(kernel_dir, "grad_cp_*_*"))
        file_count = len(vp_gradients)
        filesize = list(set([os.path.getsize(_i) for _i in vp_gradients]))
        if len(filesize) != 1:
            msg = ("The grad_cp_*_* files in '%s' do not all have "
                "identical size.") % kernel_dir
            raise LASIFCommandLineException(msg)
        filesize = filesize[0]
        # Now loop over all model directories until a fitting one if found.
        for model_dir in model_directories:
            # Use the lambda parameter files. One could also use any of the
            # others.
            lambda_files = glob(os.path.join(model_dir, "lambda*"))
            if len(lambda_files) != file_count:
                continue
            l_filesize = list(
                set([os.path.getsize(_i) for _i in lambda_files]))
            if len(l_filesize) != 1 or l_filesize[0] != filesize:
                continue
            model_boxfile = os.path.join(model_dir, "boxfile")
            if not os.path.exists(model_boxfile):
                continue
            boxfile_found = True
            boxfile = model_boxfile
        if boxfile_found is not True:
            msg = ("Could not find a suitable boxfile for the kernel stored "
                "in '%s'. Please either copy a suitable one to this "
                "directory or add a model with the same dimension to LASIF. "
                "LASIF will then be able to figure out the dimensions of it.")
            raise LASIFCommandLineException(msg)
        shutil.copyfile(boxfile, os.path.join(kernel_dir, "boxfile"))

    handler = ses3d_models.RawSES3DModelHandler(kernel_dir,
        model_type="kernel")
    handler.rotation_axis = proj.domain["rotation_axis"]
    handler.rotation_angle_in_degree = proj.domain["rotation_angle"]

    while True:
        print handler
        print ""

        inp = raw_input("Enter 'COMPONENT DEPTH' ('quit' to exit): ")
        if inp.lower() == "quit":
            break
        try:
            component, depth = inp.split()
        except:
            continue

        try:
            handler.parse_component(component)
        except:
            continue
        handler.plot_depth_slice(component, int(depth))


def lasif_plot_model(args):
    """
    Usage lasif plot_model MODEL_NAME
    """
    from lasif import ses3d_models

    if len(args) != 1:
        msg = "MODEL_NAME must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    model_name = args[0]

    proj = _find_project_root(".")

    model_dir = proj.get_model_dict()[model_name]
    handler = ses3d_models.RawSES3DModelHandler(model_dir)
    handler.rotation_axis = proj.domain["rotation_axis"]
    handler.rotation_angle_in_degree = proj.domain["rotation_angle"]

    while True:
        print handler
        print ""

        inp = raw_input("Enter 'COMPONENT DEPTH' ('quit' to exit): ")
        if inp.lower() == "quit":
            break
        try:
            component, depth = inp.split()
        except:
            continue

        try:
            handler.parse_component(component)
        except:
            continue
        handler.plot_depth_slice(component, float(depth))


def lasif_event_info(args):
    """
    Usage: lasif event_info EVENT_NAME

    Prints information about the given event.
    """
    from lasif.utils import table_printer
    if len(args) != 1:
        msg = "EVENT_NAME must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    event_name = args[0]

    proj = _find_project_root(".")
    try:
        event_dict = proj.get_event_info(event_name)
    except Exception as e:
        raise LASIFCommandLineException(str(e))

    print "Earthquake with %.1f %s at %s" % (event_dict["magnitude"],
        event_dict["magnitude_type"], event_dict["region"])
    print "\tLatitude: %.3f, Longitude: %.3f, Depth: %.1f km" % (
        event_dict["latitude"], event_dict["longitude"],
        event_dict["depth_in_km"])
    print "\t%s UTC" % str(event_dict["origin_time"])

    stations = proj.get_stations_for_event(event_name)
    print "\nStation and waveform information available at %i stations:\n" \
        % len(stations)
    header = ["id", "latitude", "longitude", "elevation", "local depth"]
    keys = sorted(stations.keys())
    data = [[key, stations[key]["latitude"], stations[key]["longitude"],
        stations[key]["elevation"], stations[key]["local_depth"]]
        for key in keys]
    table_printer(header, data)


def lasif_plot_stf(args):
    """
    Usage: lasif plot_stf ITERATION_NAME

    Convenience function to have a look at how a source time function will
    look for any iteration.
    """
    import lasif.visualization
    proj = _find_project_root(".")

    if len(args) != 1:
        msg = ("ITERATION_NAME must be given.")
        raise LASIFCommandLineException(msg)
    iteration_name = args[0]

    iteration = proj._get_iteration(iteration_name)
    stf = iteration.get_source_time_function()

    lasif.visualization.plot_tf(stf["data"], stf["delta"])


def lasif_generate_input_files(args):
    """
    Usage: lasif generate_input_files ITERATION_NAME EVENT_NAME SIMULATION_TYPE

    TYPE denotes the type of simulation to run. Available types are
        * "normal_simulation"
        * "adjoint_forward"
        * "adjoint_reverse"

    Generates the input files for one event for a specific iteration.
    """
    proj = _find_project_root(".")

    if len(args) != 3:
        msg = ("ITERATION_NAME, EVENT_NAME, and SIMULATION_TYPE must be given."
            " No other arguments allowed.")
        raise LASIFCommandLineException(msg)
    iteration_name = args[0]
    event_name = args[1]
    simulation_type = args[2].lower()

    # Assert a correct simulation type.
    simulation_types = ("normal_simulation", "adjoint_forward",
            "adjoint_reverse")
    if simulation_type not in simulation_types:
        msg = "Invalid simulation type '%s'. Available types: %s" % \
            (simulation_type, ", ".join(simulation_types))
        raise LASIFCommandLineException(msg)

    simulation_type = simulation_type.replace("_", " ")

    proj.generate_input_files(iteration_name, event_name, simulation_type)


def lasif_init_project(args):
    """
    Usage: lasif init_project FOLDER_PATH

    Creates a new LASIF project at FOLDER_PATH. FOLDER_PATH must not exist
    yet and will be created.
    """
    if len(args) != 1:
        msg = "FOLDER_PATH must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    folder_path = args[0]
    if os.path.exists(folder_path):
        msg = "The given FOLDER_PATH already exists. It must not exist yet."
        raise LASIFCommandLineException(msg)
    folder_path = os.path.abspath(folder_path)
    try:
        os.makedirs(folder_path)
    except:
        msg = "Failed creating directory %s. Permissions?" % folder_path
        raise LASIFCommandLineException(msg)

    Project(project_root_path=folder_path,
        init_project=os.path.basename(folder_path))

    print("Initialized project in: \n\t%s" % folder_path)


def lasif_finalize_adjoint_sources(args):
    """
    Usage: lasif finalize_adjoint_sources ITERATION_NAME EVENT_NAME

    Finalizes the adjoint sources for the given iteration and event.
    """
    if len(args) != 2:
        msg = "ITERATION_NAME and EVENT_NAME must be given."
        raise LASIFCommandLineException(msg)

    proj = _find_project_root(".")

    iteration_name = args[0]
    event_name = args[1]

    proj.finalize_adjoint_sources(iteration_name, event_name)


def lasif_launch_misfit_gui(args):
    """
    Usage: lasif launch_misfit_gui ITERATION_NAME EVENT_NAME

    Launches the Misfit GUI for the given iteration and event.
    """
    if len(args) != 2:
        msg = "ITERATION_NAME and EVENT_NAME must be given."
        raise LASIFCommandLineException(msg)

    proj = _find_project_root(".")

    iteration_name = args[0]
    event_name = args[1]

    events = proj.get_event_dict()
    if event_name not in events:
        msg = "Event '%s' not found in project." % event_name
        raise LASIFCommandLineException(msg)

    from lasif.misfit_gui import MisfitGUI
    from lasif.window_manager import MisfitWindowManager
    from lasif.adjoint_src_manager import AdjointSourceManager

    iterator = proj.data_synthetic_iterator(event_name, iteration_name)

    long_iteration_name = "ITERATION_%s" % iteration_name

    window_directory = os.path.join(proj.paths["windows"], event_name,
        long_iteration_name)
    ad_src_directory = os.path.join(proj.paths["adjoint_sources"], event_name,
        long_iteration_name)
    window_manager = MisfitWindowManager(window_directory, long_iteration_name,
        event_name)
    adj_src_manager = AdjointSourceManager(ad_src_directory)

    event = proj.get_event(event_name)
    MisfitGUI(event, iterator, proj, window_manager, adj_src_manager)


def lasif_create_new_iteration(args):
    """
    Usage: lasif create_new_iteration ITERATION_NAME SOLVER_NAME

    Creates a new iteration XML file.

    ITERATION_NAME determines the name of the iteration. Should always start
    with a number, so that sorting works correctly.

    SOLVER_NAME is the name of the waveform solver to use for this iteration.
    Currently available: "SES3D_4_0"
    """
    if len(args) != 2:
        msg = "ITERATION_NAME and SOLVER_NAME must be given."
        raise LASIFCommandLineException(msg)
    iteration_name = args[0]
    solver_name = args[1]

    proj = _find_project_root(".")

    proj.create_new_iteration(iteration_name, solver_name)


def lasif_list_iterations(args):
    """
    Usage: lasif list_iterations

    Returns a list of all iterations for this project.
    """
    iterations = _find_project_root(".").get_iteration_dict().keys()
    print("%i Iteration%s in project:" % (len(iterations),
        "s" if len(iterations) > 1 else ""))
    for iteration in sorted(iterations):
        print ("\t%s" % iteration)


def lasif_iteration_info(args):
    """
    Usage: lasif iteration_info ITERATION_NAME

    Prints information about the given event.
    """
    from lasif.iteration_xml import Iteration

    if len(args) != 1:
        msg = "ITERATION_NAME must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    iteration_name = args[0]

    proj = _find_project_root(".")
    iterations = proj.get_iteration_dict()
    if iteration_name not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
            "a list of all available iterations.") % iteration_name
        raise LASIFCommandLineException(msg)

    iteration = Iteration(iterations[iteration_name])
    print iteration


def lasif_remove_empty_coordinate_entries(args):
    """
    Usage: lasif remove_empty_coordinate_entries

    Removes all empty coordinate entries in the inventory cache. This is
    useful if you want to try to download coordinates again.
    """
    from lasif.tools.inventory_db import reset_coordinate_less_stations

    proj = _find_project_root(".")
    reset_coordinate_less_stations(proj.paths["inv_db_file"])

    print "SUCCESS"


def lasif_preprocess_data(args):
    """
    Usage: lasif preprocess_data ITERATION_NAME

    Preprocesses all currently available data for a given iteration.
    """
    if len(args) != 1:
        msg = "ITERATION_NAME must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    iteration_name = args[0]

    proj = _find_project_root(".")
    iterations = proj.get_iteration_dict()
    if iteration_name not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
            "a list of all available iterations.") % iteration_name
        raise LASIFCommandLineException(msg)

    proj.preprocess_data(iteration_name)


def lasif_plot_selected_windows(args):
    """
    Usage: lasif plot_selected_windows ITERATION_NAME EVENT_NAME

    Plots the automatically selected windows for a given Iteration and event.
    """
    if len(args) != 2:
        msg = "ITERATION_NAME and EVENT_NAME must be given."
        raise LASIFCommandLineException(msg)

    proj = _find_project_root(".")

    iteration_name = args[0]
    event_name = args[1]

    events = proj.get_event_dict()
    if event_name not in events:
        msg = "Event '%s' not found in project." % event_name
        raise LASIFCommandLineException(msg)

    iterator = proj.data_synthetic_iterator(event_name, iteration_name)

    long_iteration_name = "ITERATION_%s" % iteration_name

    for i in iterator:
        print i



def lasif_generate_dummy_data(args):
    """
    Usage: lasif generate_dummy_data

    Generates some random example event and waveforms. Useful for debugging,
    testing, and following the tutorial.
    """
    import inspect
    from lasif import rotations
    from lasif.adjoint_sources.utils import get_dispersed_wavetrain
    import numpy as np
    import obspy

    if len(args):
        msg = "No arguments allowed."
        raise LASIFCommandLineException(msg)

    proj = _find_project_root(".")

    # Use a seed to make it somewhat predictable.
    random.seed(34235234)
    # Create 5 events.
    d = proj.domain["bounds"]
    b = d["boundary_width_in_degree"] * 1.5
    event_count = 8
    for _i in xrange(8):
        lat = random.uniform(d["minimum_latitude"] + b,
            d["maximum_latitude"] - b)
        lon = random.uniform(d["minimum_longitude"] + b,
            d["maximum_longitude"] - b)
        depth_in_m = random.uniform(d["minimum_depth_in_km"],
            d["maximum_depth_in_km"]) * 1000.0
        # Rotate the coordinates.
        lat, lon = rotations.rotate_lat_lon(lat, lon,
            proj.domain["rotation_axis"], proj.domain["rotation_angle"])
        time = obspy.UTCDateTime(random.uniform(
            obspy.UTCDateTime(2008, 1, 1).timestamp,
            obspy.UTCDateTime(2013, 1, 1).timestamp))

        # The moment tensor. XXX: Make sensible values!
        values = [-3.3e+18, 1.43e+18, 1.87e+18, -1.43e+18, -2.69e+17,
            -1.77e+18]
        random.shuffle(values)

        mrr = values[0]
        mtt = values[1]
        mpp = values[2]
        mrt = values[3]
        mrp = values[4]
        mtp = values[5]
        mag = random.uniform(5, 7)
        scalar_moment = 3.661e+25

        event_name = os.path.join(proj.paths["events"],
            "dummy_event_%i.xml" % (_i + 1))

        cat = obspy.core.event.Catalog(events=[
            obspy.core.event.Event(
                event_type="earthquake",
                origins=[obspy.core.event.Origin(
                    latitude=lat, longitude=lon, depth=depth_in_m, time=time)],
                magnitudes=[obspy.core.event.Magnitude(
                    mag=mag, magnitude_type="Mw")],
                focal_mechanisms=[obspy.core.event.FocalMechanism(
                    moment_tensor=obspy.core.event.MomentTensor(
                        scalar_moment=scalar_moment,
                        tensor=obspy.core.event.Tensor(m_rr=mrr, m_tt=mtt,
                            m_pp=mpp, m_rt=mrt, m_rp=mrp, m_tp=mtp)))])])
        cat.write(event_name, format="quakeml", validate=False)
    print "Generated %i random events." % event_count

    # Update the folder structure.
    proj.update_folder_structure()

    names_taken = []

    def _get_random_name(length):
        while True:
            ret = ""
            for i in xrange(length):
                ret += chr(int(random.uniform(ord("A"), ord("Z"))))
            if ret in names_taken:
                continue
            names_taken.append(ret)
            break
        return ret

    # Now generate 30 station coordinates. Use a land-sea mask included in
    # basemap and loop until thirty stations on land are found.
    from mpl_toolkits.basemap import _readlsmask
    from obspy.core.util.geodetics import gps2DistAzimuth
    ls_lon, ls_lat, ls_mask = _readlsmask()
    stations = []
    # Do not use an infinite loop. One could choose a region with no land.
    for i in xrange(10000):
        if len(stations) >= 30:
            break
        lat = random.uniform(d["minimum_latitude"] + b,
            d["maximum_latitude"] - b)
        lon = random.uniform(d["minimum_longitude"] + b,
            d["maximum_longitude"] - b)
        # Rotate the coordinates.
        lat, lon = rotations.rotate_lat_lon(lat, lon,
            proj.domain["rotation_axis"], proj.domain["rotation_angle"])
        if not ls_mask[np.abs(ls_lat - lat).argmin()][
                np.abs(ls_lon - lon).argmin()]:
            continue
        stations.append({"latitude": lat, "longitude": lon,
            "network": "XX", "station": _get_random_name(3)})

    if not len(stations):
        msg = "Could not create stations. Pure ocean region?"
        raise ValueError(msg)

    # Create a RESP file for every channel.
    resp_file_temp = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), os.path.pardir, "tools",
        "RESP.template_file")
    with open(resp_file_temp, "rt") as open_file:
        resp_file_template = open_file.read()

    for station in stations:
        for component in ["E", "N", "Z"]:
            filename = os.path.join(proj.paths["resp"], "RESP.%s.%s.%s.BE%s" %
                (station["network"], station["station"], "", component))
            with open(filename, "wt") as open_file:
                open_file.write(resp_file_template.format(
                    station=station["station"], network=station["network"],
                    channel="BH%s" % component))

    print "Generated %i RESP files." % (30 * 3)

    def _empty_sac_trace():
        """
        Helper function to create and empty SAC header.
        """
        sac_dict = {}
        # floats = -12345.8
        floats = ["a", "mag", "az", "baz", "cmpaz", "cmpinc", "b", "depmax",
            "depmen", "depmin", "dist", "e", "evdp", "evla", "evlo", "f",
            "gcarc", "o", "odelta", "stdp", "stel", "stla", "stlo", "t0", "t1",
            "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "unused10",
            "unused11", "unused12", "unused6", "unused7", "unused8", "unused9",
            "user0", "user1", "user2", "user3", "user4", "user5", "user6",
            "user7", "user8", "user9", "xmaximum", "xminimum", "ymaximum",
            "yminimum"]
        sac_dict.update({key: -12345.0 for key in floats})
        # Integers: -12345
        integers = ["idep", "ievreg", "ievtype", "iftype", "iinst", "imagsrc",
            "imagtyp", "iqual", "istreg", "isynth", "iztype", "lcalda",
            "lovrok", "nevid", "norid", "nwfid"]
        sac_dict.update({key: -12345 for key in integers})
        # Strings: "-12345  "
        strings = ["ka", "kdatrd", "kevnm", "kf", "kinst", "ko", "kt0", "kt1",
            "kt2", "kt3", "kt4", "kt5", "kt6", "kt7", "kt8", "kt9",
            "kuser0", "kuser1", "kuser2"]

        sac_dict.update({key: "-12345  " for key in strings})

        # Header version
        sac_dict["nvhdr"] = 6
        # Data is evenly spaced
        sac_dict["leven"] = 1
        # And a positive polarity.
        sac_dict["lpspol"] = 1

        tr = obspy.Trace()
        tr.stats.sac = obspy.core.AttribDict(sac_dict)
        return tr

    events = proj.get_all_events()
    # Now loop over all events and create SAC file for them.
    for _i, event in enumerate(events):
        lat, lng = event.origins[0].latitude, event.origins[0].longitude
        # Get the distance to each events.
        for station in stations:
            # Add some perturbations.
            distance_in_km = gps2DistAzimuth(lat, lng, station["latitude"],
                station["longitude"])[0] / 1000.0
            a = random.uniform(3.9, 4.1)
            b = random.uniform(0.9, 1.1)
            c = random.uniform(0.9, 1.1)
            body_wave_factor = random.uniform(0.095, 0.015)
            body_wave_freq_scale = random.uniform(0.45, 0.55)
            distance_in_km = random.uniform(0.99 * distance_in_km, 1.01 *
                distance_in_km)
            _, u = get_dispersed_wavetrain(dw=0.001,
                distance=distance_in_km, t_min=0, t_max=900, a=a, b=b, c=c,
                body_wave_factor=body_wave_factor,
                body_wave_freq_scale=body_wave_freq_scale)
            for component in ["E", "N", "Z"]:
                tr = _empty_sac_trace()
                tr.data = u
                tr.stats.network = station["network"]
                tr.stats.station = station["station"]
                tr.stats.location = ""
                tr.stats.channel = "BH%s" % component
                tr.stats.sac.stla = station["latitude"]
                tr.stats.sac.stlo = station["longitude"]
                tr.stats.sac.stdp = 0.0
                tr.stats.sac.stel = 0.0
                path = os.path.join(proj.paths["data"],
                    "dummy_event_%i" % (_i + 1), "raw")
                if not os.path.exists(path):
                    os.makedirs(path)
                tr.write(os.path.join(path, "%s.%s..BH%s.sac" %
                    (station["network"], station["station"], component)),
                    format="sac")
    print "Generated %i waveform files." % (30 * 3 * len(events))


def main():
    """
    Main entry point for the script collection.

    Essentially just dispatches the different commands to the corresponding
    functions. Also provides some convenience functionality like error catching
    and printing the help.
    """
    # Get all functions in this script starting with "lasif_".
    fcts = {fct_name[len(FCT_PREFIX):]: fct for (fct_name, fct) in
            globals().iteritems()
            if fct_name.startswith(FCT_PREFIX) and hasattr(fct, "__call__")}
    # Parse args.
    args = sys.argv[1:]
    # Print help if none are given.
    if not args:
        _print_generic_help(fcts)
        sys.exit(1)
    fct_name = args[0]
    further_args = args[1:]
    # Print help if given function is not known.
    if fct_name not in fcts:
        _print_generic_help(fcts)
        sys.exit(1)
    if further_args and further_args[0] == "help":
        print_fct_help(fct_name)
        sys.exit(0)
    try:
        fcts[fct_name](further_args)
    except LASIFCommandLineException as e:
        print(colorama.Fore.YELLOW + ("Error: %s\n" % e.message) +
            colorama.Style.RESET_ALL)
        print_fct_help(fct_name)
        sys.exit(1)
    except Exception as e:
        print(colorama.Fore.RED)
        traceback.print_exc()
        print(colorama.Style.RESET_ALL)


def _print_generic_help(fcts):
    """
    Small helper function printing a generic help message.
    """
    print("Usage: lasif FUNCTION PARAMETERS\n")
    print("Available functions:")
    for name in sorted(fcts.keys()):
        print("\t%s" % name)
    print("\nTo get help for a specific function type")
    print("\tlasif FUNCTION help")


def print_fct_help(fct_name):
    """
    Prints a function specific help string. Essentially just prints the
    docstring of the function which is supposed to be formatted in a way thats
    useful as a console help message.
    """
    doc = globals()[FCT_PREFIX + fct_name].__doc__
    doc = doc.strip()
    print(doc)
