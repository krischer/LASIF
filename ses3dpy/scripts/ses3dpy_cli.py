#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The main SES3DPy console script.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
FCT_PREFIX = "ses3d_"

import colorama
import os
import sys

from ses3dpy.project import Project


class SES3DCommandLineException(Exception):
    pass


def _find_project_root(folder):
    """
    Will search upwards from the given folder until a folder containing a
    SES3DPy root structure is found. The absolute path to the root is returned.
    """
    max_folder_depth = 10
    folder = folder
    for _ in xrange(max_folder_depth):
        if os.path.exists(os.path.join(folder, "simulation_domain.xml")):
            return Project(os.path.abspath(folder))
        folder = os.path.join(folder, os.path.pardir)
    msg = "Not inside a SES3D project."
    raise SES3DCommandLineException(msg)


def ses3d_plot_domain(args):
    """
    Usage ses3dpy plot_domain

    Plots the project's domain on a map.
    """
    proj = _find_project_root(".")
    proj.plot_domain()


def ses3d_plot_events(args):
    """
    Usage ses3dpy plot_events

    Plots all events.
    """
    proj = _find_project_root(".")
    proj.plot_events()


def ses3d_info(args):
    """
    Usage ses3dpy info

    Print information about the current project.
    """
    proj = _find_project_root(".")
    print(proj)


def ses3d_init_project(args):
    """
    Usage: ses3dpy init_project FOLDER_PATH

    Creates a new SES3DPy project at FOLDER_PATH. FOLDER_PATH must not exist
    yet and will be created.
    """
    if len(args) != 1:
        msg = "FOLDER_PATH must be given. No other arguments allowed."
        raise SES3DCommandLineException(msg)
    folder_path = args[0]
    if os.path.exists(folder_path):
        msg = "The given FOLDER_PATH already exists. It must not exist yet."
        raise SES3DCommandLineException(msg)
    folder_path = os.path.abspath(folder_path)
    try:
        os.makedirs(folder_path)
    except:
        msg = "Failed creating directory %s. Permissions?" % folder_path
        raise SES3DCommandLineException(msg)
    # Now create all the subfolders.
    os.mkdir(os.path.join(folder_path, "EVENTS"))
    os.mkdir(os.path.join(folder_path, "DATA"))
    os.mkdir(os.path.join(folder_path, "SYNTHETICS"))
    os.mkdir(os.path.join(folder_path, "MODELS"))

    xml_file = (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<domain>\n"
        "  <name>{project_name}</name>\n"
        "  <description></description>\n"
        "  <domain_bounds>\n"
        "    <minimum_longitude>-20.0</minimum_longitude>\n"
        "    <maximum_longitude>20.0</maximum_longitude>\n"
        "    <minimum_latitude>-20.0</minimum_latitude>\n"
        "    <maximum_latitude>20.0</maximum_latitude>\n"
        "    <minimum_depth_in_km>0.0</minimum_depth_in_km>\n"
        "    <maximum_depth_in_km>200.0</maximum_depth_in_km>\n"
        "  </domain_bounds>\n"
        "  <domain_rotation>\n"
        "    <rotation_axis_x>1.0</rotation_axis_x>\n"
        "    <rotation_axis_y>1.0</rotation_axis_y>\n"
        "    <rotation_axis_z>1.0</rotation_axis_z>\n"
        "    <rotation_angle_in_degree>35.0</rotation_angle_in_degree>\n"
        "  </domain_rotation>\n"
        "</domain>\n")

    with open(os.path.join(folder_path, "simulation_domain.xml"), "wt") as \
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
    # Get all functions in this script starting with "ses3d_".
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
    except SES3DCommandLineException as e:
        print(colorama.Fore.RED + ("Error: %s\n" % e.message) +
            colorama.Style.RESET_ALL)
        print_fct_help(fct_name)
        sys.exit(1)


def _print_generic_help(fcts):
    """
    Small helper function printing a generic help message.
    """
    print("Usage: ses3dpy FUNCTION PARAMETERS\n")
    print("Available functions:")
    for name in fcts.iterkeys():
        print("\t%s" % name)
    print("\nTo get help for a specific function type")
    print("\tses3dpy FUNCTION help")


def print_fct_help(fct_name):
    """
    Prints a function specific help string. Essentially just prints the
    docstring of the function which is supposed to be formatted in a way thats
    useful as a console help message.
    """
    doc = globals()[FCT_PREFIX + fct_name].__doc__
    doc = doc.strip()
    print(doc)
