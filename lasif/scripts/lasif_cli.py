#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The main LASIF console script.

It is important to import necessary things at the method level to make
importing this file as fast as possible. Otherwise using the command line
interface feels sluggish and slow.


All functions starting with "lasif_" will automatically be available as
subcommands to the main "lasif" command. A decorator to determine the category
of a function is provided.

The help for every function can be accessed either via

lasif help CMD_NAME

or

lasif CMD_NAME --help


The former will be converted to the later and each subcommand is responsible
for handling the --help argument.


Each function will be passed a parser and args. It is the function author's
responsibility to add any arguments and call

parser.parse_args(args)

when done. See the existing functions for some examples. This architecture
should scale fairly well and makes it trivial to add new methods.


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import os
from lasif import LASIFException

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import difflib
import shutil
import sys
import traceback

import colorama

from lasif.project import Project


FCT_PREFIX = "lasif_"


# Documentation for the subcommand groups. This will appear in the CLI
# documentation.
COMMAND_GROUP_DOCS = {
    "Data Acquisition": (
        "These functions are used to acquire and archive different types of "
        "data usually from webservices."
    ),
    "Event Management": (
        "Function helping in organzing the earthquakes inside a LASIF "
        "project."
    ),
    "Iteration Management": (
        "Functions dealing with one or more iterations inside a LASIF "
        "project."
    ),
    "Misc": (
        "All functions that do not fit in one of the other categories."
    ),
    "Misc": (
        "All functions that do not fit in one of the other categories."
    ),
    "Plotting": (
        "Functions producing pictures."
    ),
    "Project Management": (
        "Functions dealing with LASIF projects as a whole."
    )
}


def command_group(group_name):
    """
    Decorator to be able to logically group commands.
    """
    def wrapper(func):
        func.group_name = group_name
        return func
    return wrapper


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


@command_group("Plotting")
def lasif_plot_domain(parser, args):
    """
    Plot the project's domain on a map.
    """
    import matplotlib.pyplot as plt
    parser.parse_args(args)

    proj = _find_project_root(".")
    proj.plot_domain()
    plt.show()


@command_group("Plotting")
def lasif_plot_event(parser, args):
    """
    Plot a single event including stations on a map.
    """
    import matplotlib.pyplot as plt
    parser.add_argument("event_name", help="name of the event to plot")
    event_name = parser.parse_args(args).event_name

    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda x: sys.exit(0))

    proj = _find_project_root(".")
    proj.plot_event(event_name)
    plt.show()


@command_group("Plotting")
def lasif_plot_station(parser, args):
    """
    Plots data for a single station and event.

    Useful for interactive data discovery.
    """
    import matplotlib.pyplot as plt
    parser.add_argument("station_id", help="name of the station to plot")
    parser.add_argument("event_name", help="name of the event to plot")
    station_id = parser.parse_args(args).station_id
    event_name = parser.parse_args(args).event_name

    proj = _find_project_root(".")
    proj.plot_station(station_id, event_name)
    plt.show()


@command_group("Plotting")
def lasif_plot_events(parser, args):
    """
    Plot all events.

    type can be one of:
        * ``map`` (default) - a map view of the events
        * ``depth`` - a depth distribution histogram
        * ``time`` - a time distribution histogram
    """
    import matplotlib.pyplot as plt
    parser.add_argument("--type", default="map", choices=["map", "depth",
                                                          "time"],
                        help="the type of plot. "
                        "``map``: beachballs on a map, "
                        "``depth``: depth distribution histogram, "
                        "``time``: time distribution histogram")
    plot_type = parser.parse_args(args).type

    fig = plt.figure()
    fig.canvas.mpl_connect("close_event", lambda x: sys.exit(0))

    proj = _find_project_root(".")
    proj.plot_events(plot_type)
    plt.show()


@command_group("Plotting")
def lasif_plot_raydensity(parser, args):
    """
    Plot a binned raycoverage plot for all events.
    """
    parser.parse_args(args)

    proj = _find_project_root(".")
    proj.plot_raydensity()


@command_group("Data Acquisition")
def lasif_add_spud_event(parser, args):
    """
    Add an event from the IRIS SPUD webservice to the project.
    """
    parser.add_argument("url", help="any SPUD momenttensor URL")
    url = parser.parse_args(args).url

    from lasif.scripts.iris2quakeml import iris2quakeml

    proj = _find_project_root(".")
    iris2quakeml(url, proj.paths["events"])


@command_group("Project Management")
def lasif_info(parser, args):
    """
    Print a summary of the project.
    """
    parser.parse_args(args)

    proj = _find_project_root(".")
    print(proj)


@command_group("Data Acquisition")
def lasif_download_waveforms(parser, args):
    """
    Download waveforms for one event.
    """
    parser.add_argument("event_name", help="name of the event")
    event_name = parser.parse_args(args).event_name

    proj = _find_project_root(".")
    if event_name not in proj.events:
        msg = "Event '%s' not found." % event_name
        raise LASIFCommandLineException(msg)

    from lasif.download_helpers import downloader

    time = proj.events[event_name]["origin_time"]
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

    downloader.download_waveforms(
        min_lat, max_lat, min_lng, max_lng,
        domain["rotation_axis"], domain["rotation_angle"], starttime, endtime,
        proj.config["download_settings"]["arclink_username"],
        channel_priority_list=channel_priority_list, logfile=logfile,
        download_folder=download_folder, waveform_format="mseed")


@command_group("Data Acquisition")
def lasif_download_stations(parser, args):
    """
    Download station files for one event.
    """
    parser.add_argument("event_name", help="name of the event")
    event_name = parser.parse_args(args).event_name

    proj = _find_project_root(".")
    if event_name not in proj.events:
        msg = "Event '%s' not found." % event_name
        raise LASIFCommandLineException(msg)

    from lasif.download_helpers import downloader

    # Fix the start- and endtime to ease download file grouping
    event_info = proj.events[event_name]
    starttime = event_info["origin_time"] - \
        proj.config["download_settings"]["seconds_before_event"]
    endtime = event_info["origin_time"] + \
        proj.config["download_settings"]["seconds_after_event"]
    time = starttime + (endtime - starttime) * 0.5

    # Get all channels.
    channels = proj._get_waveform_cache_file(event_name, "raw").get_values()
    channels = [{
        "channel_id": _i["channel_id"], "network": _i["network"],
        "station": _i["station"], "location": _i["location"],
        "channel": _i["channel"], "starttime": starttime, "endtime": endtime}
        for _i in channels]
    channels_to_download = []

    # Filter for channel not actually available
    for channel in channels:
        if proj.has_station_file(channel["channel_id"], time):
            continue
        channels_to_download.append(channel)

    downloader.download_stations(
        channels_to_download, proj.paths["resp"],
        proj.paths["station_xml"], proj.paths["dataless_seed"],
        logfile=os.path.join(proj.paths["logs"], "station_download_log.txt"),
        arclink_user=proj.config["download_settings"]["arclink_username"],
        get_station_filename_fct=proj.get_station_filename)


@command_group("Event Management")
def lasif_list_events(parser, args):
    """
    Print a list of all events in the project.
    """
    parser.parse_args(args)

    from lasif.tools.prettytable import PrettyTable
    proj = _find_project_root(".")
    print("%i event%s in project:" % (len(proj.events),
          "s" if len(proj.events) != 1 else ""))
    tab = PrettyTable(["Event Name", "Lat/Lng/Depth(km)/Mag",
                       "# raw/preproc/synth"])
    tab.align["Event Name"] = "l"
    for event in sorted(proj.events.keys()):
        ev = proj.events[event]
        count = proj.get_filecounts_for_event(event)
        tab.add_row([
            event, "%6.1f / %6.1f / %3i / %3.1f" % (
                ev["latitude"], ev["longitude"], int(ev["depth_in_km"]),
                ev["magnitude"]),
            "%4i / %5i / %4i" % (
                count["raw_waveform_file_count"],
                count["preprocessed_waveform_file_count"],
                count["synthetic_waveform_file_count"])])
    print tab


@command_group("Project Management")
def lasif_list_models(parser, args):
    """
    Print a list of all models in the project.
    """
    parser.parse_args(args)

    models = _find_project_root(".").get_model_dict()
    print("%i model%s in project:" % (len(models), "s" if len(models) != 1
          else ""))
    for model in sorted(models.keys()):
        print ("\t%s" % model)


@command_group("Plotting")
def lasif_plot_kernel(parser, args):
    """
    Work in progress.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("event_name", help="name of the event")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    event_name = args.event_name

    from glob import glob
    from lasif import ses3d_models

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
            msg = (
                "Could not find a suitable boxfile for the kernel stored "
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

        inp = raw_input("Enter 'COMPONENT DEPTH' "
                        "('quit/exit' to exit): ").strip()
        if inp.lower() in ["quit", "q", "exit", "leave"]:
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


@command_group("Plotting")
def lasif_plot_model(parser, args):
    """
    Plot a SES3D model.
    """
    parser.add_argument("model_name", help="name of the model")
    model_name = parser.parse_args(args).model_name

    from lasif import ses3d_models

    proj = _find_project_root(".")

    model_dir = proj.get_model_dict()[model_name]
    handler = ses3d_models.RawSES3DModelHandler(model_dir)
    handler.rotation_axis = proj.domain["rotation_axis"]
    handler.rotation_angle_in_degree = proj.domain["rotation_angle"]

    while True:
        print handler
        print ""

        inp = raw_input("Enter 'COMPONENT DEPTH' "
                        "('quit/exit' to exit): ").strip()
        if inp.lower() in ["quit", "q", "exit", "leave"]:
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


@command_group("Plotting")
def lasif_plot_wavefield(parser, args):
    """
    Plots a SES3D wavefield.
    """
    parser.add_argument("iteration_name", help="name_of_the_iteration")
    parser.add_argument("event_name", help="name of the event")
    args = parser.parse_args(args)

    from lasif import ses3d_models

    proj = _find_project_root(".")

    event_name = args.event_name
    iteration_name = args.iteration_name
    long_iteration_name = "ITERATION_%s" % iteration_name

    wavefield_dir = os.path.join(proj.paths["wavefields"], event_name,
                                 long_iteration_name)

    if not os.path.exists(wavefield_dir) or not os.listdir(wavefield_dir):
        msg = "No data available for event and iteration combination."
        raise LASIFCommandLineException(msg)

    handler = ses3d_models.RawSES3DModelHandler(
        wavefield_dir, model_type="wavefield")
    handler.rotation_axis = proj.domain["rotation_axis"]
    handler.rotation_angle_in_degree = proj.domain["rotation_angle"]

    while True:
        print handler
        print ""

        inp = raw_input("Enter 'COMPONENT DEPTH' "
                        "('quit/exit' to exit): ").strip()
        if inp.lower() in ["quit", "q", "exit", "leave"]:
            break
        try:
            component, timestep, depth = inp.split()
        except:
            continue

        component = component = "%s %s" % (component, timestep)

        try:
            handler.parse_component(component)
        except:
            continue
        handler.plot_depth_slice(component, float(depth))


@command_group("Event Management")
def lasif_event_info(parser, args):
    """
    Print information about a single event.
    """
    parser.add_argument("event_name", help="name of the event")
    event_name = parser.parse_args(args).event_name

    from lasif.utils import table_printer

    proj = _find_project_root(".")
    if event_name not in proj.events:
        msg = "Event '%s' not found in project." % event_name
        raise LASIFCommandLineException(msg)

    event_dict = proj.events[event_name]

    print "Earthquake with %.1f %s at %s" % (
        event_dict["magnitude"], event_dict["magnitude_type"],
        event_dict["region"])
    print "\tLatitude: %.3f, Longitude: %.3f, Depth: %.1f km" % (
        event_dict["latitude"], event_dict["longitude"],
        event_dict["depth_in_km"])
    print "\t%s UTC" % str(event_dict["origin_time"])

    try:
        stations = proj.get_stations_for_event(event_name)
    except LASIFException:
        stations = {}
    print "\nStation and waveform information available at %i stations:\n" \
        % len(stations)
    header = ["id", "latitude", "longitude", "elevation_in_m", "local depth"]
    keys = sorted(stations.keys())
    data = [[
        key, stations[key]["latitude"], stations[key]["longitude"],
        stations[key]["elevation_in_m"], stations[key]["local_depth_in_m"]]
        for key in keys]
    table_printer(header, data)


@command_group("Plotting")
def lasif_plot_stf(parser, args):
    """
    Plot the source time function for one iteration.
    """
    import lasif.visualization

    parser.add_argument("iteration_name", help="name of the iteration")
    iteration_name = parser.parse_args(args).iteration_name

    proj = _find_project_root(".")

    iteration = proj._get_iteration(iteration_name)
    pp = iteration.get_process_params()
    freqmin = pp["highpass"]
    freqmax = pp["lowpass"]

    stf = iteration.get_source_time_function()

    # Ignore lots of potential warnings with some plotting functionality.
    lasif.visualization.plot_tf(stf["data"], stf["delta"], freqmin=freqmin,
                                freqmax=freqmax)


@command_group("Iteration Management")
def lasif_generate_input_files(parser, args):
    """
    Generate the input files for the waveform solver.

    TYPE denotes the type of simulation to run. Available types are
        * "normal_simulation"
        * "adjoint_forward"
        * "adjoint_reverse"
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("event_name", help="name of the event")
    parser.add_argument("--simulation_type",
                        choices=("normal_simulation", "adjoint_forward",
                                 "adjoint_reverse"),
                        default="normal_simulation",
                        help="type of simulation to run")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    event_name = args.event_name
    simulation_type = args.simulation_type

    proj = _find_project_root(".")
    simulation_type = simulation_type.replace("_", " ")
    proj.generate_input_files(iteration_name, event_name, simulation_type)


@command_group("Project Management")
def lasif_init_project(parser, args):
    """
    Create a new project.
    """
    parser.add_argument("folder_path", help="where to create the project")
    folder_path = parser.parse_args(args).folder_path

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


@command_group("Iteration Management")
def lasif_finalize_adjoint_sources(parser, args):
    """
    Finalize the adjoint sources.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("event_name", help="name of the event")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    event_name = args.event_name

    proj = _find_project_root(".")
    proj.finalize_adjoint_sources(iteration_name, event_name)


@command_group("Iteration Management")
def lasif_launch_misfit_gui(parser, args):
    """
    Launch the misfit GUI.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("event_name", help="name of the event")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    event_name = args.event_name

    proj = _find_project_root(".")

    if event_name not in proj.events:
        msg = "Event '%s' not found in project." % event_name
        raise LASIFCommandLineException(msg)

    from lasif.misfit_gui import MisfitGUI
    from lasif.window_manager import MisfitWindowManager
    from lasif.adjoint_src_manager import AdjointSourceManager

    iterator = proj.data_synthetic_iterator(event_name, iteration_name)

    long_iteration_name = "ITERATION_%s" % iteration_name

    window_directory = os.path.join(proj.paths["windows"],
                                    event_name, long_iteration_name)
    ad_src_directory = os.path.join(proj.paths["adjoint_sources"],
                                    event_name, long_iteration_name)
    window_manager = MisfitWindowManager(window_directory, long_iteration_name,
                                         event_name)
    adj_src_manager = AdjointSourceManager(ad_src_directory)

    event = proj.events[event_name]
    iteration = proj._get_iteration(iteration_name)

    MisfitGUI(event, iterator, proj, window_manager, adj_src_manager,
              iteration)


@command_group("Iteration Management")
def lasif_create_new_iteration(parser, args):
    """
    Create a new iteration.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("min_period", type=float,
                        help="the minimum period of the iteration")
    parser.add_argument("max_period", type=float,
                        help="the maximum period of the iteration")
    parser.add_argument("solver_name", help="name of the solver",
                        choices=("SES3D_4_1", "SES3D_2_0",
                                 "SPECFEM3D_CARTESIAN"))
    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    solver_name = args.solver_name
    min_period = args.min_period
    max_period = args.max_period
    if min_period >= max_period:
        msg = "min_period needs to be smaller than max_period."
        raise LASIFCommandLineException(msg)

    proj = _find_project_root(".")
    proj.create_new_iteration(iteration_name, solver_name, min_period,
                              max_period)


@command_group("Iteration Management")
def lasif_create_successive_iteration(parser, args):
    """
    Create an iteration based on an existing one.

    It will take all settings in one iteration and transfers them to another
    iteration. Any comments will be deleted.
    """
    parser.add_argument("existing_iteration",
                        help="name of the existing iteration")
    parser.add_argument("new_iteration", help="name of the new iteration")
    args = parser.parse_args(args)
    existing_iteration_name = args.existing_iteration
    new_iteration_name = args.new_iteration

    from lasif.iteration_xml import Iteration

    # Get the old iteration
    proj = _find_project_root(".")
    iterations = proj.get_iteration_dict()
    if existing_iteration_name not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
               "a list of all available iterations.") % \
            existing_iteration_name
        raise LASIFCommandLineException(msg)
    existing_iteration = Iteration(iterations[existing_iteration_name])

    # Clone the old iteration, delete any comments and change the name.
    existing_iteration.comments = []
    existing_iteration.iteration_name = new_iteration_name
    print existing_iteration


@command_group("Iteration Management")
def lasif_migrate_windows(parser, args):
    """
    Migrates windows from one iteration to the next.
    """
    parser.add_argument("from_iteration",
                        help="iteration containing windows")
    parser.add_argument("to_iteration", help="iteration windows will "
                                             "migratet to")
    args = parser.parse_args(args)

    from_iteration = args.from_iteration
    to_iteration = args.to_iteration

    proj = _find_project_root(".")

    # Get some information about the iteration.
    iterations = proj.get_iteration_dict()
    if from_iteration not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
               "a list of all available iterations.") % from_iteration
        raise LASIFCommandLineException(msg)
    if to_iteration not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
               "a list of all available iterations.") % to_iteration
        raise LASIFCommandLineException(msg)

    # After all checks have passed, import the necessary modules.
    from lasif.iteration_xml import Iteration
    from lasif.window_manager import MisfitWindowManager
    from lasif.adjoint_src_manager import AdjointSourceManager

    from_it = Iteration(iterations[from_iteration])
    to_it = Iteration(iterations[to_iteration])

    long_iteration_name_from = "ITERATION_%s" % from_iteration
    long_iteration_name_to = "ITERATION_%s" % to_iteration

    # Get a list of windows that are in both, the new and the old iteration.
    events = list(set(from_it.events.keys()).intersection(
        set(to_it.events.keys())))

    # Loop over events and migrate the windows.
    for event_name in events:
        # Managers from.
        window_directory_from = os.path.join(
            proj.paths["windows"], event_name, long_iteration_name_from)
        ad_src_directory_from = os.path.join(
            proj.paths["adjoint_sources"], event_name,
            long_iteration_name_from)
        window_manager_from = MisfitWindowManager(
            window_directory_from, long_iteration_name_from, event_name)
        adj_src_manager_from = AdjointSourceManager(ad_src_directory_from)

        # Managers to.
        window_directory_to = os.path.join(
            proj.paths["windows"], event_name, long_iteration_name_to)
        ad_src_directory_to = os.path.join(
            proj.paths["adjoint_sources"], event_name, long_iteration_name_to)
        window_manager_to = MisfitWindowManager(
            window_directory_to, long_iteration_name_to, event_name)
        adj_src_manager_to = AdjointSourceManager(ad_src_directory_to)

        # Loop over all new data.
        iterator = proj.data_synthetic_iterator(event_name, to_iteration)
        for data_set in iterator:
            for component in ["Z", "N", "E"]:
                data = data_set["data"].select(component=component)[0]
                synth = data_set["synthetics"].select(channel=component)[0]
            raise NotImplementedError


@command_group("Iteration Management")
def lasif_list_iterations(parser, args):
    """
    Print a list of all iterations in the project.
    """
    parser.parse_args(args)

    iterations = _find_project_root(".").get_iteration_dict().keys()

    print("%i iteration%s in project:" % (len(iterations),
          "s" if len(iterations) != 1 else ""))
    for iteration in sorted(iterations):
        print ("\t%s" % iteration)


@command_group("Iteration Management")
def lasif_iteration_info(parser, args):
    """
    Print information about a single iteration.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    iteration_name = parser.parse_args(args).iteration_name

    from lasif.iteration_xml import Iteration

    proj = _find_project_root(".")
    iterations = proj.get_iteration_dict()
    if iteration_name not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
               "a list of all available iterations.") % iteration_name
        raise LASIFCommandLineException(msg)

    iteration = Iteration(iterations[iteration_name])
    print iteration


@command_group("Project Management")
def lasif_remove_empty_coordinate_entries(parser, args):
    """
    Remove all empty coordinate entries in the inventory cache.

    This is useful if you want to try to download coordinates again.
    """
    parser.parse_args(args)

    from lasif.tools.inventory_db import reset_coordinate_less_stations

    proj = _find_project_root(".")
    reset_coordinate_less_stations(proj.paths["inv_db_file"])

    print "SUCCESS"


@command_group("Iteration Management")
def lasif_preprocess_data(parser, args):
    """
    Launch data preprocessing.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    events = args.events if args.events else None

    proj = _find_project_root(".")

    # Check if the iteration name is valid.
    iterations = proj.get_iteration_dict()
    if iteration_name not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
               "a list of all available iterations.") % iteration_name
        raise LASIFCommandLineException(msg)

    # Check if the event ids are valid.
    if events:
        for event_name in events:
            if event_name not in proj.events:
                msg = "Event '%s' not found." % event_name
                raise LASIFCommandLineException(msg)

    proj.preprocess_data(iteration_name, events)


@command_group("Iteration Management")
def lasif_plot_q_model(parser, args):
    """
    Plots the Q model for a given iteration.
    """
    import matplotlib.pyplot as plt
    parser.add_argument("iteration_name", help="name of iteration")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name

    proj = _find_project_root(".")

    # Check if the iteration name is valid.
    iterations = proj.get_iteration_dict()
    if iteration_name not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
               "a list of all available iterations.") % iteration_name
        raise LASIFCommandLineException(msg)
    proj.plot_Q_model(iteration_name)
    plt.show()


@command_group("Plotting")
def lasif_plot_selected_windows(parser, args):
    """
    Plot the selected windows.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("event_name", help="name of the event")
    args = parser.parse_args(args)

    iteration_name = args.iteration_name
    event_name = args.event_name

    from lasif.window_selection import select_windows, plot_windows
    proj = _find_project_root(".")

    if event_name not in proj.events:
        msg = "Event '%s' not found in project." % event_name
        raise LASIFCommandLineException(msg)

    iterator = proj.data_synthetic_iterator(event_name, iteration_name)
    event_info = proj.events[event_name]

    iteration = proj._get_iteration(iteration_name)
    process_params = iteration.get_process_params()
    minimum_period = 1.0 / process_params["lowpass"]
    maximum_period = 1.0 / process_params["highpass"]

    output_folder = proj.get_output_folder(
        "Selected_Windows_Iteration_%s__%s" % (event_name, iteration_name))

    for i in iterator:
        for component in ["Z", "N", "E"]:
            try:
                data = i["data"].select(component=component)[0]
            except IndexError:
                continue
            synthetics = i["synthetics"].select(component=component)[0]
            windows = select_windows(
                data, synthetics,
                event_info["latitude"], event_info["longitude"],
                event_info["depth_in_km"], i["coordinates"]["latitude"],
                i["coordinates"]["longitude"], minimum_period, maximum_period)
            plot_windows(
                data, synthetics, windows, maximum_period,
                filename=os.path.join(output_folder,
                                      "windows_%s.pdf" % data.id))
    print "Done. Written output to folder %s." % output_folder


@command_group("Project Management")
def lasif_validate_data(parser, args):
    """
    Validate the data currently in the project.

    This commands walks through all available data and checks it for validity.
    It furthermore does some sanity checks to detect common problems. These
    should be fixed.

    By default is only checks some things. A full check is recommended but
    potentially takes a very long time.

    Things the command does:

    Event files:
        * Validate against QuakeML 1.2 scheme.
        * Make sure they contain at least one origin, magnitude and focal
          mechanism object.
        * Check for duplicate ids amongst all QuakeML files.
        * Some simply sanity checks so that the event depth is reasonable and
          the moment tensor values as well. This is rather fragile and mainly
          intended to detect values specified in wrong units.
    """
    parser.add_argument(
        "--station_file_availability",
        help="asserts that all waveform files have corresponding station "
        "files. Very slow.",
        action="store_true")
    parser.add_argument(
        "--raypaths", help="assert that all raypaths are within the "
        "set boundaries. Very slow.", action="store_true")
    parser.add_argument(
        "--waveforms", help="asserts that waveforms for one event have only "
        "a single location and channel type. Fast.", action="store_true")

    parser.add_argument("--full", help="run all validations.",
                        action="store_true")

    args = parser.parse_args(args)
    full_check = args.full
    station_file_availability = args.station_file_availability
    raypaths = args.raypaths
    waveforms = args.waveforms

    # If full check, check everything.
    if full_check:
        station_file_availability = True
        raypaths = True
        waveforms = True

    proj = _find_project_root(".")
    proj.validate_data(station_file_availability=station_file_availability,
                       raypaths=raypaths, waveforms=waveforms)


@command_group("Iteration Management")
def lasif_iteration_status(parser, args):
    """
    Query the current status of an iteration.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    iteration_name = parser.parse_args(args).iteration_name

    proj = _find_project_root(".")
    status = proj.get_iteration_status(iteration_name)

    file_count = len(proj._get_all_raw_waveform_files_for_iteration(
                     iteration_name))

    if not status["stations_in_iteration_that_do_not_exist"]:
        file_status = "All necessary files available."
    else:
        file_status = ("{count} waveform files specified in the iteration "
                       "are not available."
                       .format(count=len(status[
                           "stations_in_iteration_that_do_not_exist"])))
    if not status["channels_not_yet_preprocessed"]:
        processing_status = "All files are preprocessed."
    else:
        processing_status = ("{proc_files} out of {file_count} files still "
                             "require preprocessing.".format(
                                 proc_files=len(status[
                                     "channels_not_yet_preprocessed"]),
                                 file_count=file_count))
    s_stat = status["synthetic_data_missing"]
    if not s_stat:
        synthetics_status = "All required synthetic files available."
    else:
        synthetics_status = \
            "Missing synthetics for {count} event{p}:\n\t\t{events}".format(
                count=len(s_stat),
                p="s" if len(s_stat) != 1 else "",
                events="\n\t\t".join(sorted([
                    "%s (for %i stations)" % (key, len(value))
                    for key, value in s_stat.iteritems()])))

    print(
        "Iteration Name: {iteration_name}\n"
        "\t{file_status}\n"
        "\t{processing_status}\n"
        "\t{synthetics_status}".format(
            iteration_name=iteration_name,
            file_status=file_status,
            processing_status=processing_status,
            synthetics_status=synthetics_status))


def lasif_tutorial(parser, args):
    """
    Open the tutorial in a webbrowser.
    """
    parser.parse_args(args)

    import webbrowser
    webbrowser.open("http://krischer.github.io/LASIF/")


def lasif_calculate_constant_q_model(parser, args):
    """
    Calculate a constant Q model useable by SES3D.
    """
    from lasif.tools import Q_discrete

    parser.add_argument("min_period", type=float,
                        help="minimum period for the constant frequency band")
    parser.add_argument("max_period", type=float,
                        help="maximum period for the constant frequency band")
    args = parser.parse_args(args)

    weights, relaxation_times, = Q_discrete.calculate_Q_model(
        N=3,
        f_min=1.0 / args.max_period,
        f_max=1.0 / args.min_period,
        iterations=10000,
        initial_temperature=0.1,
        cooling_factor=0.9998)
    print "Weights: %s" % ", ".join([str(i) for i in weights])
    print "Relaxation Times: %s" % ", ".join([str(i) for i in weights])


def lasif_debug(parser, args):
    """
    Print information LASIF can gather from a list of files.
    """
    from lasif.project import LASIFException

    parser.add_argument(
        "files", help="filenames to print debug information about", nargs="+")
    files = parser.parse_args(args).files
    proj = _find_project_root(".")

    for filename in files:
        filename = os.path.relpath(filename)
        if not os.path.exists(filename):
            print("{red}File '{f}' does not exist.{reset}\n".format(
                f=filename, red=colorama.Fore.RED,
                reset=colorama.Style.RESET_ALL))
            continue
        print("{green}File '{f}':{reset}".format(
            f=filename, green=colorama.Fore.GREEN,
            reset=colorama.Style.RESET_ALL))

        try:
            info = proj.get_debug_information_for_file(filename)
        except LASIFException as e:
            info = "Error: %s" % e.message

        print info
        print ""


@command_group("Misc")
def lasif_serve(parser, args):
    """
    Launches the LASIF webinterface.
    """
    parser.add_argument("--port", default=8008, type=int,
                        help="Port of the webserver.")

    parser.add_argument("--nobrowser", help="Do not open a webbrowser.",
                        action="store_true")
    parser.add_argument("--debug", help="Turn on debugging. Implies "
                                        "'--nobrowser'.",
                        action="store_true")
    args = parser.parse_args(args)
    port = args.port
    nobrowser = args.nobrowser
    debug = args.debug

    if debug:
        nobrowser = True

    project = _find_project_root(".")

    if nobrowser is False:
        import webbrowser
        import threading

        threading.Timer(1.0,
            lambda: webbrowser.open("http://localhost:%i" % port)).start()

    from lasif.webinterface.server import serve
    serve(project, port=port, debug=debug)


def _get_cmd_description(fct):
    """
    Convenience function extracting the first line of a docstring.
    """
    try:
        return fct.__doc__.strip().split("\n")[0].strip()
    except:
        return ""


def _print_generic_help(fcts):
    """
    Small helper function printing a generic help message.
    """
    print(80 * "#")
    header = ("{default_style}LASIF - Large Scale Seismic "
              "{inverted_style}Inversion"
              "{default_style} Framework{reset_style}".format(
                  default_style=colorama.Style.BRIGHT + colorama.Fore.WHITE +
                  colorama.Back.BLACK,
                  inverted_style=colorama.Style.BRIGHT + colorama.Fore.BLACK +
                  colorama.Back.WHITE,
                  reset_style=colorama.Style.RESET_ALL))
    print "\t" + header
    print "\thttp://krischer.github.io/LASIF"
    print(80 * "#")
    print("\n{cmd}usage: lasif [--help] COMMAND [ARGS]{reset}\n".format(
        cmd=colorama.Style.BRIGHT + colorama.Fore.RED,
        reset=colorama.Style.RESET_ALL))

    # Group the functions. Functions with no group will be placed in the group
    # "Misc".
    fct_groups = {}
    for fct_name, fct in fcts.iteritems():
        group_name = fct.group_name if hasattr(fct, "group_name") else "Misc"
        fct_groups.setdefault(group_name, {})
        fct_groups[group_name][fct_name] = fct

    # Print in a grouped manner.
    for group_name in sorted(fct_groups.iterkeys()):
        print("{0:=>25s} Functions".format(" " + group_name))
        current_fcts = fct_groups[group_name]
        for name in sorted(current_fcts.keys()):
            print("%s  %32s: %s%s%s" % (colorama.Fore.YELLOW, name,
                  colorama.Fore.BLUE,
                  _get_cmd_description(fcts[name]),
                  colorama.Style.RESET_ALL))
    print("\nTo get help for a specific function type")
    print("\tlasif help FUNCTION  or\n\tlasif FUNCTION --help")


def _get_argument_parser(fct):
    """
    Helper function to create a proper argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="lasif %s" % fct.func_name.replace("lasif_", ""),
        description=_get_cmd_description(fct))
    return parser


def _get_functions():
    """
    Get a list of all CLI functions defined in this file.
    """
    # Get all functions in this script starting with "lasif_".
    fcts = {fct_name[len(FCT_PREFIX):]: fct for (fct_name, fct) in
            globals().iteritems()
            if fct_name.startswith(FCT_PREFIX) and hasattr(fct, "__call__")}
    return fcts


def main():
    """
    Main entry point for the LASIF command line interface.

    Essentially just dispatches the different commands to the corresponding
    functions. Also provides some convenience functionality like error catching
    and printing the help.
    """
    fcts = _get_functions()
    # Parse args.
    args = sys.argv[1:]

    # Print the generic help/introduction.
    if not args or args == ["help"] or args == ["--help"]:
        _print_generic_help(fcts)
        sys.exit(1)

    # Use lowercase to increase tolerance.
    fct_name = args[0].lower()

    further_args = args[1:]
    # Map "lasif help CMD" to "lasif CMD --help"
    if fct_name == "help":
        if further_args and further_args[0] in fcts:
            fct_name = further_args[0]
            further_args = ["--help"]
        else:
            sys.stderr.write("lasif: Invalid command. See 'lasif --help'.\n")
            sys.exit(1)

    # Unknown function.
    if fct_name not in fcts:
        sys.stderr.write("lasif: '{fct_name}' is not a LASIF command. See "
                         "'lasif --help'.\n".format(fct_name=fct_name))
        # Attempt to fuzzy match commands.
        close_matches = sorted(difflib.get_close_matches(fct_name, fcts.keys(),
                                                         n=4))
        if len(close_matches) == 1:
            sys.stderr.write("\nDid you mean this?\n\t{match}\n".format(
                match=close_matches[0]))
        elif close_matches:
            sys.stderr.write(
                "\nDid you mean one of these?\n\t{matches}\n".format(
                    matches="\n\t".join(close_matches)))

        sys.exit(1)

    # Create a parser and pass it to the single function.
    parser = _get_argument_parser(fcts[fct_name])

    # Now actually call the function.
    try:
        fcts[fct_name](parser, further_args)
    except LASIFCommandLineException as e:
        print(colorama.Fore.YELLOW + ("Error: %s\n" % e.message) +
              colorama.Style.RESET_ALL)
        sys.exit(1)
    except Exception as e:
        print(colorama.Fore.RED)
        traceback.print_exc()
        print(colorama.Style.RESET_ALL)
