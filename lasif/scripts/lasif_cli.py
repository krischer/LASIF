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
import lasif
from lasif import api
from lasif.api import LASIFCommandLineException
from lasif.tools.query_gcmt_catalog import get_subset_of_events

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import colorama
import difflib
import pathlib
import sys
import traceback
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from mpi4py import MPI


# Try to disable the ObsPy deprecation warnings. This makes LASIF work with
# the latest ObsPy stable and the master.
try:
    # It only exists for certain ObsPy versions.
    from obspy.core.util.deprecation_helpers import ObsPyDeprecationWarning
except:
    pass
else:
    warnings.filterwarnings("ignore", category=ObsPyDeprecationWarning)


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


def mpi_enabled(func):
    """
    Decorator to mark function with mpi capabilities.
    """
    func._is_mpi_enabled = True
    return func


def split(container, count):
    """
    Simple and elegant function splitting a container into count
    equal chunks.

    Order is not preserved but for the use case at hand this is
    potentially an advantage as data sitting in the same folder thus
    have a higher at being processed at the same time thus the disc
    head does not have to jump around so much. Of course very
    architecture dependent.
    """
    return [container[_i::count] for _i in range(count)]


@command_group("Plotting")
def lasif_plot_domain(parser, args):
    """
    Plot the project's domain on a map.
    """
    parser.add_argument("--save", help="Save the plot in a file",
                        action="store_true")
    args = parser.parse_args(args)
    save = args.save

    api.plot_domain(lasif_root=".", save=save)


@command_group("Misc")
def lasif_shell(parser, args):
    """
    Drops you into a shell with an active communicator instance.
    """
    args = parser.parse_args(args)

    comm = api.find_project_comm(".")
    print("LASIF shell, 'comm' object is available in the local namespace.\n")
    print(comm)
    from IPython import embed
    embed(display_banner=False)


@command_group("Plotting")
def lasif_plot_event(parser, args):
    """
    Plot a single event including stations on a map.
    """
    parser.add_argument("--save", help="Saves the plot in a file",
                        action="store_true")
    parser.add_argument("event_name", help="name of the event to plot")
    parser.add_argument("--weight_set_name", help="for stations to be "
                                                  "color coded as a function "
                                                  "of their respective "
                                                  "weights", default=None)
    args = parser.parse_args(args)

    api.plot_event(lasif_root=".", event_name=args.event_name,
                   weight_set_name=args.weight_set_name,
                   save=args.save)


@command_group("Plotting")
def lasif_plot_events(parser, args):
    """
    Plot all events. This might need an extension to iterations.

    type can be one of:
        * ``map`` (default) - a map view of the events
        * ``depth`` - a depth distribution histogram
        * ``time`` - a time distribution histogram
    """
    parser.add_argument("--type", default="map", choices=["map", "depth",
                                                          "time"],
                        help="the type of plot. "
                        "``map``: beachballs on a map, "
                        "``depth``: depth distribution histogram, "
                        "``time``: time distribution histogram")
    parser.add_argument("--iteration", help="Plot all events for an "
                                            "iteration", default=None)
    parser.add_argument("--save", help="Saves the plot in a file",
                        action="store_true")
    args = parser.parse_args(args)

    api.plot_events(lasif_root=".", type=args.type, iteration=args.iteration,
                    save=args.save)


@command_group("Plotting")
def lasif_plot_raydensity(parser, args):
    """
    Plot a binned raycoverage plot for all events.
    """
    parser.add_argument("--plot_stations", help="also plot the stations",
                        action="store_true")
    args = parser.parse_args(args)

    api.plot_raydensity(lasif_root=".", plot_stations=args.plot_stations)


@command_group("Plotting")
def lasif_plot_section(parser, args):
    """
    Plot a binned section plot of the processed data for an event.
    """
    parser.add_argument("event_name", help="name of the event to plot")
    parser.add_argument("--num_bins", default=1, type=int,
                        help="number of bins to be used for binning the "
                             "event-station offsets")
    parser.add_argument("--traces_per_bin", default=500, type=int,
                        help="number of traces per bin")
    args = parser.parse_args(args)
    event_name = args.event_name
    traces_per_bin = args.traces_per_bin
    num_bins = args.num_bins

    comm = api.find_project_comm(".")
    comm.visualizations.plot_section(event_name=event_name, num_bins=num_bins,
                                     traces_per_bin=traces_per_bin)


@command_group("Data Acquisition")
def lasif_add_spud_event(parser, args):
    """
    Add an event from the IRIS SPUD webservice to the project.
    """
    parser.add_argument("url", help="any SPUD momenttensor URL")
    args = parser.parse_args(args)
    url = args.url

    from lasif.scripts.iris2quakeml import iris2quakeml

    comm = api.find_project_comm(".")
    iris2quakeml(url, comm.project.paths["eq_data"])


# # PERSONAL USE
# @command_group("Data Acquisition")
# def lasif_write_events_to_xml(parser, args):
#     """
#     Writes a collection of event xmls.
#     """
#     args = parser.parse_args(args)
#     comm = api.find_project_comm(".")
#     import pyasdf
#
#     output_folder = comm.project.get_output_folder(
#         type="EventXMLs", tag="event")
#
#     for event in comm.events.list():
#         event_filename = \
#             comm.waveforms.get_asdf_filename(event, data_type="raw")
#         with pyasdf.ASDFDataSet(event_filename, mode="r") as ds:
#             cat = ds.events
#             cat.write(os.path.join(output_folder, event + ".xml"),
#                       format="QuakeML")
#
#     print(f"You can find the collection of QuakeMl files in {output_folder}")


@command_group("Data Acquisition")
def lasif_add_gcmt_events(parser, args):
    """
    Selects and adds optimally distributed events from the GCMT catalog.
    """
    parser.add_argument("count", type=int,
                        help="maximum amount of events to add")
    parser.add_argument("min_magnitude", type=float,
                        help="minimum magnitude off events to add")
    parser.add_argument("max_magnitude", type=float,
                        help="maximum magnitude off events to add")
    parser.add_argument("min_distance", type=float,
                        help="The minimum acceptable distance to the next "
                             "closest event in km.")
    parser.add_argument("--min_year", default=None, type=int,
                        help="minimum year from which to add events")
    parser.add_argument("--max_year", default=None, type=int,
                        help="maximum year from which to add events")

    args = parser.parse_args(args)

    api.add_gcmt_events(lasif_root=".", count=args.count,
                        min_mag=args.min_magnitude,
                        max_mag=args.max_magnitude,
                        min_dist=args.min_distance,
                        min_year=args.min_year, max_year=args.max_year)


@command_group("Project Management")
def lasif_info(parser, args):
    """
    Print a summary of the project.
    """
    args = parser.parse_args(args)
    api.project_info(lasif_root=".")


@command_group("Data Acquisition")
def lasif_download_data(parser, args):
    """
    Download waveform and station data for one or more events.
    Can be used to download data for all events in LASIF project.
    """
    parser.add_argument("event_name", help="name of the event. Possible to add"
                        " more than one event separated by a space. If "
                        "argument is left empty. data will be downloaded "
                        "for all events", nargs="*")
    parser.add_argument("--providers", default=None,
                        type=str, nargs="+",
                        help="FDSN providers to query. Will use all known "
                             "ones if not set.")
    parser.add_argument("--downsample_data", action="store_true",
                        help="If the dataset could get too big this can"
                             " help with reducing the size."
                             " Be very careful while using this one. "
                             "Currently it changes the waveforms a bit.")
    args = parser.parse_args(args)

    api.download_data(lasif_root=".",
                      event_name=args.event_name if args.event_name else [],
                      providers=args.providers)


@command_group("Event Management")
def lasif_list_events(parser, args):
    """
    Print a list of all events in the project.
    """
    parser.add_argument("--list", help="Show only a list of events. Good for "
                                       "scripting bash.",
                        action="store_true")
    parser.add_argument("--iteration", help="Show only events related to "
                                            "a specific iteration",
                        default=None)
    args = parser.parse_args(args)

    api.list_events(lasif_root=".", list=args.list, iteration=args.iteration)


@command_group("Iteration Management")
def lasif_submit_job(parser, args):
    """
    EXPERIMENTAL: Submits event(s) to daint with salvus-flow. NO QA

    Requires all input_files and salvus-flow to be installed and
    configured.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("ranks", help="amount of ranks", type=int)
    parser.add_argument("wall_time_in_seconds", help="wall time", type=int)
    parser.add_argument("simulation_type", help="forward, "
                                                "step_length, adjoint")
    parser.add_argument("site", help="Computer to submit the job to")
    parser.add_argument("event", help="If you only want to submit selected "
                                       "events. You can input more than one "
                                       "separated by a space. If none is "
                                       "specified, all will be taken",
                        nargs="*", default=None)

    args = parser.parse_args(args)

    api.submit_job(lasif_root=".", iteration=args.iteration_name,
                   ranks=args.ranks, wall_time=args.wall_time_in_seconds,
                   simulation_type=args.simulation_type,
                   site=args.site, events=args.event if args.event else [])


@command_group("Iteration Management")
def lasif_retrieve_output(parser, args):
    """
    EXPERIMENTAL: Retrieves output from simulation, No QA.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("simulation_type", help="forward, "
                                                "step_length, adjoint")
    parser.add_argument("site", help="Computer to get output from")
    parser.add_argument("event", help="names of events you want to retrieve "
                                       "output from. If more than one, "
                                       "separate with space. If none specified"
                                       " all will be used.", nargs="*",
                        default=None)

    args = parser.parse_args(args)

    api.retrieve_output(lasif_root=".", iteration=args.iteration_name,
                        simulation_type=args.simulation_type,
                        site=args.site,
                        events=args.event if args.event else [])


@command_group("Event Management")
def lasif_event_info(parser, args):
    """
    Print information about a single event.
    """
    parser.add_argument("event_name", help="name of the event")
    parser.add_argument("-v", help="Verbose. Print all contained events.",
                        action="store_true")
    args = parser.parse_args(args)

    api.event_info(lasif_root=".", event_name=args.event_name, verbose=args.v)


# API
@command_group("Plotting")
def lasif_plot_stf(parser, args):
    """
    Plot the source time function for one iteration.
    """
    parser.add_argument("--unfiltered", help="Add this tag if you want to "
                                             "plot an unfiltered STF",
                        action="store_true")
    args = parser.parse_args(args)
    api.plot_stf(lasif_root=".")


@command_group("Iteration Management")
def lasif_generate_input_files(parser, args):
    """
    Generate input files for the forward simulation of the waveform solver.
    """
    parser.add_argument("iteration_name", help="name of the iteration ")
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")
    parser.add_argument("simulation_type", help="forward, "
                                                "step_length, adjoint",
                        default="forward")
    parser.add_argument("--weight_set_name", default=None, type=str,
                        help="Set of station and event weights,"
                             "used to scale the adjoint sources")
    parser.add_argument("--prev_iter", default=None, type=str,
                        help="Optionally specify a previous iteration"
                             "to use input files_from, only updates"
                             "the mesh file.")

    args = parser.parse_args(args)
    api.generate_input_files(lasif_root=".", iteration=args.iteration_name,
                             simulation_type=args.simulation_type,
                             events=args.events if args.events else [],
                             weight_set=args.weight_set_name if
                             args.weight_set_name else None,
                             prev_iter=args.prev_iter if args.prev_iter else
                             None)


@command_group("Project Management")
def lasif_init_project(parser, args):
    """
    Create a new project.
    """
    parser.add_argument("folder_path", help="where to create the project")
    args = parser.parse_args(args)

    api.init_project(args.folder_path)


@mpi_enabled
@command_group("Iteration Management")
def lasif_calculate_adjoint_sources(parser, args):
    """
    Calculates adjoint sources for a given iteration.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("window_set_name", help="name of the window_set")
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")

    args = parser.parse_args(args)

    api.calculate_adjoint_sources(lasif_root=".",
                                  iteration=args.iteration_name,
                                  window_set=args.window_set_name,
                                  events=args.events if args.events else [])


@mpi_enabled
@command_group("Iteration Management")
def lasif_select_windows(parser, args):
    """
    Autoselect windows for a given event and iteration combination.

    This function works with MPI. Don't use too many cores, I/O quickly
    becomes the limiting factor. It also works without MPI but then only one
    core actually does any work.
    """
    parser.add_argument("iteration", help="name of the iteration")
    parser.add_argument("window_set_name", help="name of the window_set")
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")
    args = parser.parse_args(args)
    api.select_windows(lasif_root=".", iteration=args.iteration,
                       window_set=args.window_set_name,
                       events=args.events if args.events else [])


@command_group("Iteration Management")
def lasif_gui(parser, args):
    """
    Launch the misfit GUI.
    """
    args = parser.parse_args(args)

    api.open_gui(lasif_root=".")


@command_group("Iteration Management")
def lasif_create_weight_set(parser, args):
    """
    Create a new set of event and station weights.
    """
    parser.add_argument("weight_set_name",
                        help="name of the weight set, i.e. \"A\"")

    args = parser.parse_args(args)
    api.create_weight_set(lasif_root=".", weight_set=args.weight_set_name)


@command_group("Iteration Management")
def lasif_compute_station_weights(parser, args):
    """
    Compute weights for stations. Useful when distribution is uneven.
    Weights are calculated for each event. This may take a while if you have
    many stations.
    """
    parser.add_argument("weight_set_name", help="Pick a name for the "
                                                "weight set. If the weight"
                                                "set already exists, it will"
                                                "overwrite")
    parser.add_argument("event_name", default=None,
                        help="name of event. If none is specified weights will"
                             "be calculated for all. Also possible to specify"
                             "more than one separated by a space", nargs="*")
    parser.add_argument("--iteration", default=None,
                        help="If you only want to do this for the events "
                             "specified for an iteration")
    args = parser.parse_args(args)

    api.compute_station_weights(lasif_root=".",
                                weight_set=args.weight_set_name,
                                events=args.event_name if args.event_name
                                else [],
                                iteration=args.iteration)


# API
@command_group("Iteration Management")
def lasif_get_weighting_bins(parser, args):
    """
    Compute median envelopes for the observed data in certain station bins.
    The binning is based on event-station distances.
    """
    parser.add_argument("window_set_name", help="Name of window set")
    parser.add_argument("event_name", default=None, help="Name of event",
                        nargs="*")
    parser.add_argument("--iteration", default=None, help="Take all events"
                                                          "used in a specific"
                                                          "iteration.")
    args = parser.parse_args(args)

    api.get_weighting_bins(lasif_root=".",
                           window_set=args.window_set_name,
                           events=args.event_name if args.event_name else [],
                           iteration=args.iteration)


@command_group("Iteration Management")
def lasif_set_up_iteration(parser, args):
    """
    Creates or removes directory structure for an iteration.
    """
    parser.add_argument("iteration_name",
                        help="name of the iteration, i.e. \"1\"")
    parser.add_argument(
        "--remove_dirs",
        help="Removes all directories related to the specified iteration. ",
        action="store_true")
    parser.add_argument("events", help="If you only want to submit selected "
                                       "events. You can input more than one "
                                       "separated by a space. If none is "
                                       "specified, all will be taken",
                        nargs="*", default=None)

    args = parser.parse_args(args)
    api.set_up_iteration(lasif_root=".", iteration=args.iteration_name,
                         events=args.events if args.events else [],
                         remove_dirs=args.remove_dirs)


@command_group("Iteration Management")
def lasif_write_misfit(parser, args):
    """
    """
    parser.add_argument("iteration_name", help="current iteration")
    parser.add_argument("--weight_set_name", default=None, type=str,
                        help="Set of station and event weights")
    parser.add_argument("--window_set_name", default=None, type=str,
                        help="name of the window set")
    args = parser.parse_args(args)

    api.write_misfit(lasif_root=".", iteration=args.iteration_name,
                     weight_set=args.weight_set_name,
                     window_set=args.window_set_name)


@command_group("Iteration Management")
def lasif_list_iterations(parser, args):
    """
    Creates directory structure for a new iteration.
    """
    args = parser.parse_args(args)
    api.list_iterations(lasif_root=".")


@mpi_enabled
@command_group("Iteration Management")
def lasif_compare_misfits(parser, args):
    """
    Compares the total misfit between two iterations.

    Total misfit is used regardless of the similarity of the picked windows
    from each iteration. This might skew the results but should
    give a good idea unless the windows change excessively between
    iterations.

    If windows are weighted in the calculation of the adjoint
    sources. That should translate into the calculated misfit
    value.
    """

    parser.add_argument("from_iteration",
                        help="past iteration")
    parser.add_argument("to_iteration", help="current iteration")
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")
    parser.add_argument("--weight_set_name", default=None, type=str,
                        help="Set of station and event weights")
    parser.add_argument("--print_events", help="compare misfits"
                                               " for each event",
                        action="store_true")
    args = parser.parse_args(args)

    api.compare_misfits(lasif_root=".", from_it=args.from_iteration,
                        to_it=args.to_iteration,
                        events=args.events if args.events else [],
                        weight_set=args.weight_set_name,
                        print_events=args.print_events)


# API
@command_group("Iteration Management")
def lasif_list_weight_sets(parser, args):
    """
    Print a list of all weight sets in the project.
    """
    args = parser.parse_args(args)
    api.list_weight_sets(lasif_root=".")


# API
@mpi_enabled
@command_group("Iteration Management")
def lasif_process_data(parser, args):
    """
    Launch data processing.

    This function works with MPI. Don't use too many cores, I/O quickly
    becomes the limiting factor. It also works without MPI but then only one
    core actually does any work.
    """
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")
    parser.add_argument("--iteration", help="Take all events used in "
                                            "iteration", default=None)

    args = parser.parse_args(args)
    api.process_data(lasif_root=".", events=args.events if args.events else [],
                     iteration=args.iteration)


@command_group("Plotting")
def lasif_plot_window_statistics(parser, args):
    """
    Plot the selected windows.
    """
    parser.add_argument("--save", help="Saves the plot in a file",
                        action="store_true")
    parser.add_argument("window_set_name", help="name of the window set")
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")
    parser.add_argument("--iteration", help="Take all events used in "
                                            "iteration", default=None)
    args = parser.parse_args(args)

    api.plot_window_statistics(lasif_root=".", window_set=args.window_set_name,
                               save=args.save,
                               events=args.events if args.events else [],
                               iteration=args.iteration)


@command_group("Plotting")
def lasif_plot_windows(parser, args):
    """
    Plot the selected windows.
    """
    parser.add_argument("event_name", help="name of the event")
    parser.add_argument("window_set_name", help="name of the window set")
    parser.add_argument("--distance_bins", type=int,
                        help="The number of bins on the distance axis for "
                             "the combined plot.",
                        default=500)
    args = parser.parse_args(args)
    api.plot_windows(lasif_root=".", event_name=args.event_name,
                     window_set=args.window_set_name,
                     distance_bins=args.distance_bins)


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
        "--data_and_station_file_availability",
        help="asserts that all stations have corresponding station "
        "files and all stations have waveforms. Very slow.",
        action="store_true")
    parser.add_argument(
        "--raypaths", help="assert that all raypaths are within the "
        "set boundaries. Very slow.", action="store_true")
    parser.add_argument("--full", help="run all validations.",
                        action="store_true")

    args = parser.parse_args(args)
    api.validate_data(lasif_root=".",
                      data_station_file_availability=
                      args.data_and_station_file_availability,
                      raypaths=args.raypaths,
                      full=args.full)


@command_group("Project Management")
def lasif_clean_up(parser, args):
    """
    Clean up the LASIF project.

    The required file can be created with lasif validate_data command.
    """
    parser.add_argument("clean_up_file", help="path of clean-up file")
    args = parser.parse_args(args)

    api.clean_up(lasif_root=".", clean_up_file=args.clean_up_file)


def lasif_tutorial(parser, args):
    """
    Open the tutorial in a webbrowser.
    """
    args = parser.parse_args(args)
    api.tutorial()


# @command_group("Misc")
# def lasif_serve(parser, args):
#     """
#     Launches the LASIF webinterface.
#     """
#     parser.add_argument("--port", default=8008, type=int,
#                         help="Port of the webserver.")
#
#     parser.add_argument("--nobrowser", help="Do not open a webbrowser.",
#                         action="store_true")
#     parser.add_argument("--debug", help="Turn on debugging. Implies "
#                                         "'--nobrowser'.",
#                         action="store_true")
#     parser.add_argument(
#         "--open_to_outside",
#         help="By default the website can only be opened from the current "
#              "computer. Use this argument to access it from any other "
#              "computer on the network.",
#         action="store_true")
#     args = parser.parse_args(args)
#     port = args.port
#     nobrowser = args.nobrowser
#     debug = args.debug
#     open_to_outside = args.open_to_outside
#
#     if debug:
#         nobrowser = True
#
#     comm = _find_project_comm(".")
#
#     if nobrowser is False:
#         import webbrowser
#         import threading
#
#         threading.Timer(
#             1.0, lambda: webbrowser.\
#               open("http://localhost:%i" % port)).start()
#
#     from lasif.webinterface.server import serve
#     serve(comm, port=port, debug=debug, open_to_outside=open_to_outside)


def _get_cmd_description(fct, extended=False):
    """
    Convenience function to extract the command description
    from the first line of the docstring.

    :param fct: The function.
    :param extended: If set to true, the function will return
    a formatted version of the entire docstring.
    """
    try:
        if extended:
            lines = fct.__doc__.split("\n")[:]
            stripped_list = [item[4:] for item in lines]
            return "\n".join(stripped_list) + "\n"
        else:
            return fct.__doc__.strip().split("\n")[0].strip()
    except:
        return ""


def _print_generic_help(fcts):
    """
    Small helper function printing a generic help message.
    """
    print(100 * "#")
    header = ("{default_style}LASIF - Large Scale Seismic "
              "{inverted_style}Inversion"
              "{default_style} Framework{reset_style}  [Version {version}]"
              .format(
                  default_style=colorama.Style.BRIGHT + colorama.Fore.WHITE +
                  colorama.Back.BLACK,
                  inverted_style=colorama.Style.BRIGHT + colorama.Fore.BLACK +
                  colorama.Back.WHITE,
                  reset_style=colorama.Style.RESET_ALL,
                  version=lasif.__version__))
    print("    " + header)
    print("    http://krischer.github.io/LASIF")
    print(100 * "#")
    print("\n{cmd}usage: lasif [--help] COMMAND [ARGS]{reset}\n".format(
        cmd=colorama.Style.BRIGHT + colorama.Fore.RED,
        reset=colorama.Style.RESET_ALL))

    # Group the functions. Functions with no group will be placed in the group
    # "Misc".
    fct_groups = {}
    for fct_name, fct in fcts.items():
        group_name = fct.group_name if hasattr(fct, "group_name") else "Misc"
        fct_groups.setdefault(group_name, {})
        fct_groups[group_name][fct_name] = fct

    # Print in a grouped manner.
    for group_name in sorted(fct_groups.keys()):
        print("{0:=>25s} Functions".format(" " + group_name))
        current_fcts = fct_groups[group_name]
        for name in sorted(current_fcts.keys()):
            print("%s  %32s: %s%s%s" % (colorama.Fore.YELLOW, name,
                  colorama.Fore.CYAN,
                  _get_cmd_description(fcts[name]),
                  colorama.Style.RESET_ALL))
    print("\nTo get help for a specific function type")
    print("\tlasif help FUNCTION  or\n\tlasif FUNCTION --help")


def _get_argument_parser(fct, extended=False):
    """
    Helper function to create a proper argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="lasif %s" % fct.__name__.replace("lasif_", ""),
        description=_get_cmd_description(fct, extended),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--ipdb",
        help="If true, a debugger will be launched upon encountering an "
             "exception. Requires ipdb.",
        action="store_true")

    # Exceptions. If any are missed, its not mission critical but just
    # less nice.
    exceptions = ["lasif_tutorial", "lasif_init_project",
                  "lasif_build_all_caches"]

    if fct.__name__ in exceptions:
        return parser

    return parser


def _get_functions():
    """
    Get a list of all CLI functions defined in this file.
    """
    # Get all functions in this script starting with "lasif_".
    fcts = {fct_name[len(FCT_PREFIX):]: fct for (fct_name, fct) in
            globals().items()
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

    if len(args) == 1 and args[0] == "--version":
        print("LASIF version %s" % lasif.__version__)
        sys.exit(0)

    # Print the generic help/introduction.
    if not args or args == ["help"] or args == ["--help"]:
        _print_generic_help(fcts)
        sys.exit(0)

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
                "\nDid you mean one of these?\n    {matches}\n".format(
                    matches="\n    ".join(close_matches)))

        sys.exit(1)

    func = fcts[fct_name]

    # Make sure that only MPI enabled functions are called with MPI.
    if MPI.COMM_WORLD.size > 1:
        if not hasattr(func, "_is_mpi_enabled") or \
                func._is_mpi_enabled is not True:
            if MPI.COMM_WORLD.rank != 0:
                return
            sys.stderr.write("'lasif %s' must not be called with MPI.\n" %
                             fct_name)
            return

    # Create a parser and pass it to the single function.
    if "--help" in further_args:
        parser = _get_argument_parser(func, extended=True)
    else:
        parser = _get_argument_parser(func)

    # Now actually call the function.
    try:
        func(parser, further_args)
    except LASIFCommandLineException as e:
        print(colorama.Fore.YELLOW + ("Error: %s\n" % str(e)) +
              colorama.Style.RESET_ALL)
        sys.exit(1)
    except Exception as e:
        args = parser.parse_args(further_args)
        # Launch ipdb debugger right at the exception point if desired.
        # Greatly eases debugging things. Requires ipdb to be installed.
        if args.ipdb:
            import ipdb  # NOQA
            _, _, tb = sys.exc_info()
            traceback.print_exc()
            ipdb.post_mortem(tb)
        else:
            print(colorama.Fore.RED)
            traceback.print_exc()
            print(colorama.Style.RESET_ALL)
