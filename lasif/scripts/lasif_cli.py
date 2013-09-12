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
import shutil
import sys
import traceback

import colorama

from lasif.project import Project


FCT_PREFIX = "lasif_"


def add_command_to_group(group_name):
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


@add_command_to_group("Plotting")
def lasif_plot_domain(args):
    """
    Usage: lasif plot_domain

    Plots the project's domain on a map.
    """
    proj = _find_project_root(".")
    proj.plot_domain()


@add_command_to_group("Plotting")
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


@add_command_to_group("Plotting")
def lasif_plot_events(args):
    """
    Usage: lasif plot_events [TYPE]

    Plots all events.

    TYPE can be one of:
        * ``map`` (default) - a map view of the events
        * ``depth`` - a depth distribution histogram
        * ``time`` - a time distribution histogram
    """
    valid_plot_types = ["map", "depth", "time"]

    proj = _find_project_root(".")

    if len(args) > 1:
        msg = "TYPE can be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    if not args:
        plot_type = "map"
    else:
        plot_type = args[0].lower()

    if plot_type not in valid_plot_types:
        msg = "Type '%s' not valid. Valid types: %s." % (plot_type, ", ".join(
            valid_plot_types))
        raise LASIFCommandLineException(msg)

    proj.plot_events(plot_type)


@add_command_to_group("Plotting")
def lasif_plot_raydensity(args):
    """
    Usage: lasif plot_raydensity

    Plots a binned raycoverage plot for all events.
    """
    proj = _find_project_root(".")
    proj.plot_raydensity()


@add_command_to_group("Data Acquisition")
def lasif_add_spud_event(args):
    """
    Usage: lasif add_spud_event URL

    Adds an event from the IRIS SPUD GCMT webservice to the project.

    URL is any SPUD momenttensor URL.
    """
    from lasif.scripts.iris2quakeml import iris2quakeml

    proj = _find_project_root(".")
    if len(args) != 1:
        msg = "URL must be given. No other arguments allowed."
        raise LASIFCommandLineException(msg)
    url = args[0]
    iris2quakeml(url, proj.paths["events"])


@add_command_to_group("Project Management")
def lasif_info(args):
    """
    Usage: lasif info

    Print information about the current project.
    """
    proj = _find_project_root(".")
    print(proj)


@add_command_to_group("Data Acquisition")
def lasif_download_waveforms(args):
    """
    Usage: lasif download_waveforms EVENT_NAME

    Downloads all waveform files for a given event.

    The list of possible events can be obtained with "lasif list_events". The
    files will be saved in the DATA/EVENT_NAME/raw directory.
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

    downloader.download_waveforms(
        min_lat, max_lat, min_lng, max_lng,
        domain["rotation_axis"], domain["rotation_angle"], starttime, endtime,
        proj.config["download_settings"]["arclink_username"],
        channel_priority_list=channel_priority_list, logfile=logfile,
        download_folder=download_folder, waveform_format="mseed")


@add_command_to_group("Data Acquisition")
def lasif_download_stations(args):
    """
    Usage: lasif download_stations EVENT_NAME

    Downloads all missing station data files for a given event.

    The list of possible events can be obtained with "lasif list_events". The
    files will be saved in the STATION/*.
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


@add_command_to_group("Event Management")
def lasif_list_events(args):
    """
    Usage: lasif list_events

    Returns a list of all events in the project.
    """
    events = _find_project_root(".").get_event_dict()
    print("%i event%s in project:" % (len(events), "s" if len(events) > 1
          else ""))
    for event in events.iterkeys():
        print ("\t%s" % event)


@add_command_to_group("Project Management")
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


@add_command_to_group("Plotting")
def lasif_plot_kernel(args):
    """
    Usage lasif plot_kernel ITERATION_NAME EVENT_NAME

    Work in progress.
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


@add_command_to_group("Plotting")
def lasif_plot_model(args):
    """
    Usage lasif plot_model MODEL_NAME

    Plots a SES3D model.
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


@add_command_to_group("Event Management")
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

    print "Earthquake with %.1f %s at %s" % (
        event_dict["magnitude"], event_dict["magnitude_type"],
        event_dict["region"])
    print "\tLatitude: %.3f, Longitude: %.3f, Depth: %.1f km" % (
        event_dict["latitude"], event_dict["longitude"],
        event_dict["depth_in_km"])
    print "\t%s UTC" % str(event_dict["origin_time"])

    stations = proj.get_stations_for_event(event_name)
    print "\nStation and waveform information available at %i stations:\n" \
        % len(stations)
    header = ["id", "latitude", "longitude", "elevation", "local depth"]
    keys = sorted(stations.keys())
    data = [[
        key, stations[key]["latitude"], stations[key]["longitude"],
        stations[key]["elevation"], stations[key]["local_depth"]]
        for key in keys]
    table_printer(header, data)


@add_command_to_group("Plotting")
def lasif_plot_stf(args):
    """
    Usage: lasif plot_stf ITERATION_NAME

    Plot the source time function.

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


@add_command_to_group("Iteration Management")
def lasif_generate_input_files(args):
    """
    Usage: lasif generate_input_files ITERATION_NAME EVENT_NAME SIMULATION_TYPE

    Generate the input files for one event.

    TYPE denotes the type of simulation to run. Available types are
        * "normal_simulation"
        * "adjoint_forward"
        * "adjoint_reverse"
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


@add_command_to_group("Project Management")
def lasif_init_project(args):
    """
    Usage: lasif init_project FOLDER_PATH

    Create a new LASIF project.

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


@add_command_to_group("Iteration Management")
def lasif_finalize_adjoint_sources(args):
    """
    Usage: lasif finalize_adjoint_sources ITERATION_NAME EVENT_NAME

    Finalize the adjoint sources for the given iteration and event.
    """
    if len(args) != 2:
        msg = "ITERATION_NAME and EVENT_NAME must be given."
        raise LASIFCommandLineException(msg)

    proj = _find_project_root(".")

    iteration_name = args[0]
    event_name = args[1]

    proj.finalize_adjoint_sources(iteration_name, event_name)


@add_command_to_group("Iteration Management")
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

    window_directory = os.path.join(proj.paths["windows"],
                                    event_name, long_iteration_name)
    ad_src_directory = os.path.join(proj.paths["adjoint_sources"],
                                    event_name, long_iteration_name)
    window_manager = MisfitWindowManager(window_directory, long_iteration_name,
                                         event_name)
    adj_src_manager = AdjointSourceManager(ad_src_directory)

    event = proj.get_event(event_name)
    MisfitGUI(event, iterator, proj, window_manager, adj_src_manager)


@add_command_to_group("Iteration Management")
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


@add_command_to_group("Iteration Management")
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


@add_command_to_group("Iteration Management")
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


@add_command_to_group("Project Management")
def lasif_remove_empty_coordinate_entries(args):
    """
    Usage: lasif remove_empty_coordinate_entries

    Remove all empty coordinate entries in the inventory cache.

    This is useful if you want to try to download coordinates again.
    """
    from lasif.tools.inventory_db import reset_coordinate_less_stations

    proj = _find_project_root(".")
    reset_coordinate_less_stations(proj.paths["inv_db_file"])

    print "SUCCESS"


@add_command_to_group("Iteration Management")
def lasif_preprocess_data(args):
    """
    Usage: lasif preprocess_data ITERATION_NAME (EVENT_NAME)

    Preprocesse all currently available data for a given iteration.
    """
    if len(args) != 2:
        msg = ("ITERATION_NAME and at most one EVENT_NAME must be given. No"
               " other arguments allowed.")
        raise LASIFCommandLineException(msg)
    iteration_name = args[0]
    event_id = args[1]

    proj = _find_project_root(".")
    iterations = proj.get_iteration_dict()
    if iteration_name not in iterations:
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
               "a list of all available iterations.") % iteration_name
        raise LASIFCommandLineException(msg)

    proj.preprocess_data(iteration_name, event_id)


@add_command_to_group("Plotting")
def lasif_plot_selected_windows(args):
    """
    Usage: lasif plot_selected_windows ITERATION_NAME EVENT_NAME

    Plot the automatically selected windows for a given iteration and event.
    """
    from lasif.window_selection import select_windows, plot_windows
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
    event_info = proj.get_event_info(event_name)

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


@add_command_to_group("Project Management")
def lasif_validate_data(args):
    """
    Usage lasif validate_data

    Validates all data of the current project.

    This commands walks through all available data and checks it for validity.
    It furthermore does some sanity checks to detect common problems. These
    should be fixed.

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
    if len(args):
        msg = "No arguments allowed."
        raise LASIFCommandLineException(msg)

    proj = _find_project_root(".")
    proj.validate_data()


def lasif_tutorial(args):
    """
    Usage lasif tutorial

    Opens the tutorial.
    """
    import webbrowser
    webbrowser.open("http://krischer.github.io/LASIF/")


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
    print(80 * "#" + "\n")
    header = ("{default_style}LASIF - LArge Scale {inverted_style}Inversion"
              "{default_style} Framework{reset_style}".format(
                  default_style=colorama.Style.BRIGHT + colorama.Fore.WHITE +
                  colorama.Back.BLACK,
                  inverted_style=colorama.Style.BRIGHT + colorama.Fore.BLACK +
                  colorama.Back.WHITE,
                  reset_style=colorama.Style.RESET_ALL))
    print "\t" + header
    print "\n\thttp://krischer.github.io/LASIF"
    print("\n" + 80 * "#")
    print("\nUsage: lasif FUNCTION PARAMETERS\n")

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
                  fcts[name].__doc__.strip().split("\n")[2].strip(),
                  colorama.Style.RESET_ALL))
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
