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
    Usage fwiw plot_domain

    Plots the project's domain on a map.
    """
    proj = _find_project_root(".")
    proj.plot_domain()


def fwiw_plot_events(args):
    """
    Usage fwiw plot_events

    Plots all events.
    """
    proj = _find_project_root(".")
    proj.plot_events()


def fwiw_info(args):
    """
    Usage fwiw info

    Print information about the current project.
    """
    proj = _find_project_root(".")
    print(proj)


def init_folder_structure(root_folder):
    """
    Updates or initializes a projects folder structure.
    """
    for folder in ["EVENTS", "DATA", "SYNTHETICS", "MODELS", "STATIONS",
            "LOGS"]:
        full_path = os.path.join(root_folder, folder)
        if os.path.exists(full_path):
            continue
        os.makedirs(full_path)

    station_folder = os.path.join(root_folder, "STATIONS")
    subfolders = ["SEED", "StationXML", "RESP"]
    for f in subfolders:
        folder = os.path.join(station_folder, f)
        if os.path.exists(folder):
            continue
        os.makedirs(folder)


def fwiw_update_structure(args):
    """
    Usage: fwiw update_structure

    Updates the folder structure of a project. Will create data and synthetics
    subfolders for every event.
    """
    proj = _find_project_root(".")
    init_folder_structure(proj.paths["root"])

    event_folder = proj.paths["events"]
    data_folder = proj.paths["data"]
    synth_folder = proj.paths["synthetics"]
    for event in glob.iglob(os.path.join(event_folder, "*.xml")):
        name = os.path.splitext(os.path.basename(event))[0]
        data_subfolder = os.path.join(data_folder, name)
        synth_subfolder = os.path.join(synth_folder, name)
        for f in [data_subfolder, synth_subfolder]:
            if os.path.exists(f):
                continue
            os.makedirs(f)


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

    downloader.download(min_lat, max_lat, min_lng, max_lng,
        domain["rotation_axis"], domain["rotation_angle"], starttime, endtime,
        proj.config["download_settings"]["arclink_username"],
        channel_priority_list=channel_priority_list, logfile=logfile,
        download_folder=download_folder, waveform_format="mseed")


def fwiw_list_events(args):
    """
    Usage: fwiw list_events

    Returns a list of all events in the project.
    """
    proj = _find_project_root(".")
    events = proj.get_event_dict()
    print("%i event%s in project:" % (len(events), "s" if len(events) > 1
        else ""))
    for event in events.iterkeys():
        print ("\t%s" % event)


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
    # Now create all the subfolders.
    init_folder_structure(folder_path)

    xml_file = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<fwiw_project>\n"
        "    <name>{project_name}</name>\n"
        "    <description></description>\n"
        "    <download_settings>\n"
        "        <arclink_username></arclink_username>\n"
        "        <seconds_before_event>300</seconds_before_event>\n"
        "        <seconds_after_event>3600</seconds_after_event>\n"
        "    </download_settings>\n"
        "    <domain>\n"
        "      <domain_bounds>\n"
        "        <minimum_longitude>-20.0</minimum_longitude>\n"
        "        <maximum_longitude>20.0</maximum_longitude>\n"
        "        <minimum_latitude>-20.0</minimum_latitude>\n"
        "        <maximum_latitude>20.0</maximum_latitude>\n"
        "        <minimum_depth_in_km>0.0</minimum_depth_in_km>\n"
        "        <maximum_depth_in_km>200.0</maximum_depth_in_km>\n"
        "        <boundary_width_in_degree>3.0</boundary_width_in_degree>\n"
        "      </domain_bounds>\n"
        "      <domain_rotation>\n"
        "        <rotation_axis_x>1.0</rotation_axis_x>\n"
        "        <rotation_axis_y>1.0</rotation_axis_y>\n"
        "        <rotation_axis_z>1.0</rotation_axis_z>\n"
        "        <rotation_angle_in_degree>-45.0</rotation_angle_in_degree>\n"
        "      </domain_rotation>\n"
        "    </domain>\n"
        "</fwiw_project>")

    with open(os.path.join(folder_path, "config.xml"), "wt") as \
            open_file:
        open_file.write(xml_file.format(project_name=os.path.basename(
            folder_path)))

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
    for name in fcts.iterkeys():
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
