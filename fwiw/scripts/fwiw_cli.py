#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The main FWIW console script.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
FCT_PREFIX = "fwiw_"

import colorama
import glob
import obspy
import os
import sys

from fwiw.project import Project
from fwiw.download_helpers import downloader
from fwiw.scripts.iris2quakeml import iris2quakeml
from fwiw.utils import table_printer, generate_ses3d_4_0_template
import fwiw.visualization


class FWIWCommandLineException(Exception):
    pass


def _find_project_root(folder):
    """
    Will search upwards from the given folder until a folder containing a
    FWIW root structure is found. The absolute path to the root is returned.
    """
    max_folder_depth = 10
    folder = folder
    for _ in xrange(max_folder_depth):
        if os.path.exists(os.path.join(folder, "config.xml")):
            return Project(os.path.abspath(folder))
        folder = os.path.join(folder, os.path.pardir)
    msg = "Not inside a FWIW project."
    raise FWIWCommandLineException(msg)


def fwiw_plot_domain(args):
    """
    Usage: fwiw plot_domain

    Plots the project's domain on a map.
    """
    proj = _find_project_root(".")
    proj.plot_domain()


def fwiw_plot_event(args):
    """
    Usage: fwiw plot_event EVENT_NAME

    Plots one event and raypaths on a map.
    """
    proj = _find_project_root(".")

    if len(args) != 1:
        msg = "EVENT_NAME must be given. No other arguments allowed."
        raise FWIWCommandLineException(msg)
    event_name = args[0]

    proj.plot_event(event_name)


def fwiw_plot_events(args):
    """
    Usage: fwiw plot_events

    Plots all events.
    """
    proj = _find_project_root(".")
    proj.plot_events()


def fwiw_add_spud_event(args):
    """
    Usage: fwiw add_spud_event URL

    Adds an event from the IRIS SPUD GCMT webservice to the project. URL is any
    SPUD momenttensor URL.
    """
    proj = _find_project_root(".")
    if len(args) != 1:
        msg = "URL must be given. No other arguments allowed."
        raise FWIWCommandLineException(msg)
    url = args[0]
    iris2quakeml(url, proj.paths["events"])


def fwiw_info(args):
    """
    Usage: fwiw info

    Print information about the current project.
    """
    proj = _find_project_root(".")
    print(proj)


def fwiw_download_waveforms(args):
    """
    Usage: fwiw download_waveforms EVENT_NAME

    Attempts to download all missing waveform files for a given event. The list
    of possible events can be obtained with "fwiw list_events". The files will
    be saved in the DATA/EVENT_NAME/raw directory.
    """
    proj = _find_project_root(".")
    events = proj.get_event_dict()
    if len(args) != 1:
        msg = "EVENT_NAME must be given. No other arguments allowed."
        raise FWIWCommandLineException(msg)
    event_name = args[0]
    if event_name not in events:
        msg = "Event '%s' not found." % event_name
        raise FWIWCommandLineException(msg)

    event = obspy.readEvents(events[event_name])[0]
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


def fwiw_download_stations(args):
    """
    Usage: fwiw download_stations EVENT_NAME

    Attempts to download all missing station data files for a given event. The
    list of possible events can be obtained with "fwiw list_events". The files
    will be saved in the STATION/*.
    """
    proj = _find_project_root(".")
    events = proj.get_event_dict()
    if len(args) != 1:
        msg = "EVENT_NAME must be given. No other arguments allowed."
        raise FWIWCommandLineException(msg)
    event_name = args[0]
    if event_name not in events:
        msg = "Event '%s' not found." % event_name
        raise FWIWCommandLineException(msg)

    channel_path = os.path.join(proj.paths["data"], event_name, "raw")
    if not os.path.exists(channel_path):
        msg = "The path '%s' does not exists." % channel_path
        raise FWIWCommandLineException(msg)

    channels = glob.glob(os.path.join(channel_path, "*"))
    if not channels:
        msg = "No data in folder '%s'" % channel_path
        raise FWIWCommandLineException(msg)

    downloader.download_stations(channels, proj.paths["resp"],
        proj.paths["station_xml"], proj.paths["dataless_seed"],
        logfile=os.path.join(proj.paths["logs"], "station_download_log.txt"),
        arclink_user=proj.config["download_settings"]["arclink_username"],
        has_station_file_fct=proj.has_station_file,
        get_station_filename_fct=proj.get_station_filename)


def fwiw_list_events(args):
    """
    Usage: fwiw list_events

    Returns a list of all events in the project.
    """
    events = _find_project_root(".").get_event_dict()
    print("%i event%s in project:" % (len(events), "s" if len(events) > 1
        else ""))
    for event in events.iterkeys():
        print ("\t%s" % event)


def fwiw_list_models(args):
    """
    Usage: fwiw list_models

    Returns a list of all models in the project.
    """
    models = _find_project_root(".").get_model_dict()
    print("%i model%s in project:" % (len(models), "s" if len(models) > 1
        else ""))
    for model in models.iterkeys():
        print ("\t%s" % model)


def fwiw_event_info(args):
    """
    Usage: fwiw init_project EVENT_NAME

    Prints information about the given event.
    """
    if len(args) != 1:
        msg = "EVENT_NAME must be given. No other arguments allowed."
        raise FWIWCommandLineException(msg)
    event_name = args[0]

    proj = _find_project_root(".")
    try:
        event_dict = proj.get_event_info(event_name)
    except Exception as e:
        raise FWIWCommandLineException(str(e))

    print "Earthquake with %.1f %s at %s" % (event_dict["magnitude"],
        event_dict["magnitude_type"], event_dict["region"])
    print "\tLatitude: %.3f, Longitude: %.3f, Depth: %.1f km" % (
        event_dict["latitude"], event_dict["longitude"],
        event_dict["depth_in_km"])
    print "\t%s UTC" % str(event_dict["origin_time"])

    try:
        stations = proj.get_stations_for_event(event_name)
    except Exception as e:
        raise FWIWCommandLineException(str(e))
    print "\nStation and waveform information available at %i stations:\n" \
        % len(stations)
    header = ["id", "latitude", "longitude", "elevation", "local depth"]
    keys = sorted(stations.keys())
    data = [[key, stations[key]["latitude"], stations[key]["longitude"],
        stations[key]["elevation"], stations[key]["local_depth"]]
        for key in keys]
    table_printer(header, data)


def fwiw_list_input_file_templates(args):
    """
    Usage: fwiw list_input_file_templates

    Returns a list of names with all input file templates.
    """
    proj = _find_project_root(".")
    files = glob.glob(os.path.join(proj.paths["templates"], "*.xml"))
    print "Project has %i input file template%s:" % (len(files), "s"
        if len(files) > 1 else "")
    for filename in files:
        print "\t%s" % os.path.splitext(os.path.basename(filename))[0]


def fwiw_list_stf(args):
    """
    Usage: fwiw list_stf

    Returns a list of names with all source time functions known to the
    project.
    """
    proj = _find_project_root(".")
    files = glob.glob(os.path.join(proj.paths["source_time_functions"],
        "*.py"))
    print "Project has %i defined source time function%s:" % (len(files), "s"
        if len(files) > 1 else "")
    for filename in files:
        print "\t%s" % os.path.splitext(os.path.basename(filename))[0]


def fwiw_plot_stf(args):
    """
    Usage: fwiw plot_stf SOURCE_TIME_FCT NPTS DELTA

    Convenience function to have a look at how a source time function will
    look.

    NPTS is the number of samples, and DELTA the sample interval.
    """
    proj = _find_project_root(".")

    if len(args) != 3:
        msg = ("SOURCE_TIME_FCT, NPTS, and DELTA must be given. "
            "No other arguments allowed.")
        raise FWIWCommandLineException(msg)
    stf = args[0]
    npts = int(args[1])
    delta = float(args[2])

    source_time_function = proj._get_source_time_function(stf)
    data = source_time_function(npts, delta)
    fwiw.visualization.plot_tf(data, delta)


def fwiw_generate_input_files(args):
    """
    Usage: fwiw generate_input_files EVENT INPUT_FILE_TEMPLATE TYPE SFT

    TYPE denotes the type of simulation to run. Available types are
        * "normal_simulation"
        * "adjoint_forward"
        * "adjoint_reverse"

    SFT is the name of the source time function to be used

    Generates the input files for one event.
    """
    proj = _find_project_root(".")

    if len(args) != 4:
        msg = ("EVENT, INPUT_FILE_TEMPLATE, TYPE, and SFT must be given. "
            "No other arguments allowed.")
        raise FWIWCommandLineException(msg)
    event_name = args[0]
    input_file_template = args[1]
    simulation_type = args[2].lower()
    source_time_function = args[3]

    # Assert a correct simulation type.
    simulation_types = ("normal_simulation", "adjoint_forward",
            "adjoint_reverse")
    if simulation_type not in simulation_types:
        msg = "Invalid simulation type '%s'. Available types: %s" % \
            (simulation_type, ", ".join(simulation_types))
        raise FWIWCommandLineException(msg)

    simulation_type = simulation_type.replace("_", " ")

    try:
        source_time_function = \
            proj._get_source_time_function(source_time_function)
    except Exception as e:
        raise FWIWCommandLineException(str(e))

    try:
        proj.generate_input_files(event_name, input_file_template,
            simulation_type, source_time_function)
    except Exception as e:
        raise FWIWCommandLineException(str(e))


def fwiw_generate_input_file_template(args):
    """
    Usage: fwiw generate_input_file_template SOLVER

    Generates a new input file template for the specified solver. Currently
    supported solvers: ses3d_4_0
    """
    if len(args) != 1:
        msg = "SOLVER must be given. No other arguments allowed."
        raise FWIWCommandLineException(msg)
    solver = args[0]

    SOLVERS = ["ses3d_4_0"]
    if solver not in SOLVERS:
        msg = "'%s' is not a valid solver. Valid solvers: %s" % (solver,
            ", ".join(SOLVERS))
        raise FWIWCommandLineException(msg)

    proj = _find_project_root(".")

    def xml_filename_generator(folder, name):
        for _i in xrange(100000):
            filename = "%s_template" % name
            if _i:
                filename += "_%i" % _i
            filename += "%sxml" % os.path.extsep
            filename = os.path.join(folder, filename)
            if os.path.exists(filename):
                continue
            return filename

    if solver == "ses3d_4_0":
        filename = xml_filename_generator(proj.paths["templates"], solver)
        generate_ses3d_4_0_template(filename)
        print "Created template at '%s'. Please edit it." % filename


def fwiw_init_project(args):
    """
    Usage: fwiw init_project FOLDER_PATH

    Creates a new FWIW project at FOLDER_PATH. FOLDER_PATH must not exist
    yet and will be created.
    """
    if len(args) != 1:
        msg = "FOLDER_PATH must be given. No other arguments allowed."
        raise FWIWCommandLineException(msg)
    folder_path = args[0]
    if os.path.exists(folder_path):
        msg = "The given FOLDER_PATH already exists. It must not exist yet."
        raise FWIWCommandLineException(msg)
    folder_path = os.path.abspath(folder_path)
    try:
        os.makedirs(folder_path)
    except:
        msg = "Failed creating directory %s. Permissions?" % folder_path
        raise FWIWCommandLineException(msg)

    Project(project_root_path=folder_path,
        init_project=os.path.basename(folder_path))

    print("Initialized project in: \n\t%s" % folder_path)


def main():
    """
    Main entry point for the script collection.

    Essentially just dispatches the different commands to the corresponding
    functions. Also provides some convenience functionality like error catching
    and printing the help.
    """
    # Get all functions in this script starting with "fwiw_".
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
    except FWIWCommandLineException as e:
        print(colorama.Fore.RED + ("Error: %s\n" % e.message) +
            colorama.Style.RESET_ALL)
        print_fct_help(fct_name)
        sys.exit(1)


def _print_generic_help(fcts):
    """
    Small helper function printing a generic help message.
    """
    print("Usage: fwiw FUNCTION PARAMETERS\n")
    print("Available functions:")
    for name in sorted(fcts.keys()):
        print("\t%s" % name)
    print("\nTo get help for a specific function type")
    print("\tfwiw FUNCTION help")


def print_fct_help(fct_name):
    """
    Prints a function specific help string. Essentially just prints the
    docstring of the function which is supposed to be formatted in a way thats
    useful as a console help message.
    """
    doc = globals()[FCT_PREFIX + fct_name].__doc__
    doc = doc.strip()
    print(doc)
