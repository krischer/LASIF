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
from lasif import LASIFError

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import collections
import colorama
import difflib
import itertools
import pathlib
import sys
import time
import traceback
import warnings

from mpi4py import MPI

from lasif import LASIFNotFoundError
from lasif.components.project import Project

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


class LASIFCommandLineException(Exception):
    pass


def _find_project_comm(folder):
    """
    Will search upwards from the given folder until a folder containing a
    LASIF root structure is found. The absolute path to the root is returned.
    """
    folder = pathlib.Path(folder).absolute()
    max_folder_depth = 10
    folder = folder
    for _ in range(max_folder_depth):
        if (folder / "lasif_config.toml").exists():
            return Project(folder).get_communicator()
        folder = folder.parent
    msg = "Not inside a LASIF project."
    raise LASIFCommandLineException(msg)


def _find_project_comm_mpi(folder):
    """
    Parallel version. Will open the caches for rank 0 with write access,
    caches from the other ranks can only read.

    :param folder: The folder were to start the search.
    """
    if MPI.COMM_WORLD.rank == 0:
        comm = _find_project_comm(folder)

    # Open the caches for the other ranks after rank zero has opened it to
    # allow for the initial caches to be written.
    MPI.COMM_WORLD.barrier()

    if MPI.COMM_WORLD.rank != 0:
        comm = _find_project_comm(folder)

    return comm


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
    parser.add_argument("--no_simulation_domain",
                        help="Don't plot the simulation domain",
                        action="store_false")
    args = parser.parse_args(args)
    comm = _find_project_comm(".")

    comm.visualizations.plot_domain(
        plot_simulation_domain=args.no_simulation_domain)

    import matplotlib.pyplot as plt
    plt.show()


@command_group("Misc")
def lasif_shell(parser, args):
    """
    Drops you into a shell with an active communicator instance.
    """
    args = parser.parse_args(args)

    comm = _find_project_comm(".")
    print("LASIF shell, 'comm' object is available in the local namespace.\n")
    print(comm)
    from IPython import embed
    embed(display_banner=False)


@command_group("Plotting")
def lasif_plot_event(parser, args):
    """
    Plot a single event including stations on a map.
    """
    parser.add_argument("event_name", help="name of the event to plot")
    args = parser.parse_args(args)
    event_name = args.event_name

    comm = _find_project_comm(".")
    comm.visualizations.plot_event(event_name)

    import matplotlib.pyplot as plt
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
    parser.add_argument("--type", default="map", choices=["map", "depth",
                                                          "time"],
                        help="the type of plot. "
                        "``map``: beachballs on a map, "
                        "``depth``: depth distribution histogram, "
                        "``time``: time distribution histogram")
    args = parser.parse_args(args)
    plot_type = args.type

    comm = _find_project_comm(".")
    comm.visualizations.plot_events(plot_type)

    import matplotlib.pyplot as plt
    plt.show()


@command_group("Plotting")
def lasif_plot_raydensity(parser, args):
    """
    Plot a binned raycoverage plot for all events.
    """
    parser.add_argument("--plot_stations", help="also plot the stations",
                        action="store_true")
    args = parser.parse_args(args)

    comm = _find_project_comm(".")
    comm.visualizations.plot_raydensity(plot_stations=args.plot_stations)


@command_group("Data Acquisition")
def lasif_add_spud_event(parser, args):
    """
    Add an event from the IRIS SPUD webservice to the project.
    """
    parser.add_argument("url", help="any SPUD momenttensor URL")
    args = parser.parse_args(args)
    url = args.url

    from lasif.scripts.iris2quakeml import iris2quakeml

    comm = _find_project_comm(".")
    iris2quakeml(url, comm.project.paths["eq_data"])


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

    from lasif.tools.query_gcmt_catalog import add_new_events
    comm = _find_project_comm(".")

    add_new_events(comm=comm, count=args.count,
                   min_magnitude=args.min_magnitude,
                   max_magnitude=args.max_magnitude,
                   min_year=args.min_year, max_year=args.max_year,
                   threshold_distance_in_km=args.min_distance)


@command_group("Project Management")
def lasif_info(parser, args):
    """
    Print a summary of the project.
    """
    args = parser.parse_args(args)

    comm = _find_project_comm(".")
    print(comm.project)


@command_group("Data Acquisition")
def lasif_download_data(parser, args):
    """
    Download waveform and station data for one event.
    """
    parser.add_argument("event_name", help="name of the event")
    parser.add_argument("--providers", default=None,
                        type=str, nargs="+",
                        help="FDSN providers to query. Will use all known "
                             "ones if not set.")
    args = parser.parse_args(args)
    event_name = args.event_name
    providers = args.providers

    comm = _find_project_comm(".")
    comm.downloads.download_data(event_name, providers=providers)


@command_group("Event Management")
def lasif_list_events(parser, args):
    """
    Print a list of all events in the project.
    """
    parser.add_argument("--details", help="print details of filecounts for "
                                          "all events",
                        action="store_true")
    parser.add_argument("--list", help="Show only a list of events. Good for "
                                       "scripting bash.",
                        action="store_true")
    args = parser.parse_args(args)

    from lasif.tools.prettytable import PrettyTable

    comm = _find_project_comm(".")

    if args.list and args.details:
        raise LASIFCommandLineException("--list and --details cannot both "
                                        "be specified.")

    if args.list is False:
        print("%i event%s in project:" % (comm.events.count(),
              "s" if comm.events.count() != 1 else ""))

    if args.details is True:
        tab = PrettyTable(["Event Name", "Lat/Lng/Depth(km)/Mag",
                           "# raw/preproc/synth"])
        tab.align["Event Name"] = "l"
        for event in comm.events.list():
            ev = comm.events.get(event)
            count = comm.project.get_filecounts_for_event(event)
            tab.add_row([
                event, "%6.1f / %6.1f / %3i / %3.1f" % (
                    ev["latitude"], ev["longitude"], int(ev["depth_in_km"]),
                    ev["magnitude"]),
                "%4i / %5i / %4i" % (
                    count["raw_waveform_file_count"],
                    count["preprocessed_waveform_file_count"],
                    count["synthetic_waveform_file_count"])])
        print(tab)
    elif args.list is True:
        for event in sorted(comm.events.list()):
            print(event)
    else:
        tab = PrettyTable(["Event Name", "Lat/Lng/Depth(km)/Mag"])
        tab.align["Event Name"] = "l"
        for event in comm.events.list():
            ev = comm.events.get(event)
            tab.add_row([
                event, "%6.1f / %6.1f / %3i / %3.1f" % (
                    ev["latitude"], ev["longitude"], int(ev["depth_in_km"]),
                    ev["magnitude"])])
        print(tab)


@command_group("Event Management")
def lasif_event_info(parser, args):
    """
    Print information about a single event.
    """
    parser.add_argument("event_name", help="name of the event")
    parser.add_argument("-v", help="Verbose. Print all contained events.",
                        action="store_true")
    args = parser.parse_args(args)
    event_name = args.event_name
    verbose = args.v

    comm = _find_project_comm(".")
    if not comm.events.has_event(event_name):
        msg = "Event '%s' not found in project." % event_name
        raise LASIFCommandLineException(msg)

    event_dict = comm.events.get(event_name)

    print("Earthquake with %.1f %s at %s" % (
          event_dict["magnitude"], event_dict["magnitude_type"],
          event_dict["region"]))
    print("\tLatitude: %.3f, Longitude: %.3f, Depth: %.1f km" % (
          event_dict["latitude"], event_dict["longitude"],
          event_dict["depth_in_km"]))
    print("\t%s UTC" % str(event_dict["origin_time"]))

    try:
        stations = comm.query.get_all_stations_for_event(event_name)
    except LASIFError:
        stations = {}

    if verbose:
        from lasif.utils import table_printer
        print("\nStation and waveform information available at %i "
              "stations:\n" % len(stations))
        header = ["ID", "Latitude", "Longitude", "Elevation_in_m"]
        keys = sorted(stations.keys())
        data = [[
            key, stations[key]["latitude"], stations[key]["longitude"],
            stations[key]["elevation_in_m"]]
            for key in keys]
        table_printer(header, data)
    else:
        print("\nStation and waveform information available at %i stations. "
              "Use '-v' to print them." % len(stations))


@command_group("Plotting")
def lasif_plot_stf(parser, args):
    """
    Plot the source time function for one iteration.
    """
    import lasif.visualization
    comm = _find_project_comm(".")

    freqmax = 1.0 / comm.project.preprocessing_params["highpass_period"]
    freqmin = 1.0 / comm.project.preprocessing_params["lowpass_period"]

    stf_fct = comm.project.get_project_function(
        "source_time_function")

    delta = comm.project.simulation_params["time_increment"]
    npts = comm.project.simulation_params["number_of_time_steps"]

    stf = {"delta": delta}

    stf["data"] = stf_fct(npts=npts, delta=delta, freqmin=freqmin, freqmax=freqmax)

    # Ignore lots of potential warnings with some plotting functionality.
    lasif.visualization.plot_tf(stf["data"], stf["delta"], freqmin=freqmin,
                                freqmax=freqmax)


@command_group("Iteration Management")
def lasif_generate_all_input_files(parser, args):
    """
    Generates all input files for a certain iteration.

    TYPE denotes the type of simulation to run. Available types are
        * "normal_simulation"
        * "adjoint_forward"
        * "adjoint_reverse"
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("weight_set_name", help="name of the weight set")
    parser.add_argument("--simulation_type",
                        choices=("normal_simulation", "adjoint_forward",
                                 "adjoint_reverse"),
                        default="normal_simulation",
                        help="type of simulation to run")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    weight_set_name = args.weight_set_name

    simulation_type = args.simulation_type

    comm = _find_project_comm(".")
    simulation_type = simulation_type.replace("_", " ")

    weights = comm.weights.get(weight_set_name)
    events = sorted(weights.events.keys())
    for _i, event in enumerate(events):
        print("Generating input files for event %i of %i..." % (_i + 1,
                                                                len(events)))
        comm.actions.generate_input_files(weight_set_name, iteration_name, event,
                                          simulation_type)


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

    comm = _find_project_comm(".")
    simulation_type = simulation_type.replace("_", " ")
    comm.actions.generate_input_files(iteration_name, event_name,
                                      simulation_type)


@command_group("Project Management")
def lasif_init_project(parser, args):
    """
    Create a new project.
    """
    parser.add_argument("folder_path", help="where to create the project")
    args = parser.parse_args(args)
    folder_path = pathlib.Path(args.folder_path).absolute()

    if folder_path.exists():
        msg = "The given FOLDER_PATH already exists. It must not exist yet."
        raise LASIFCommandLineException(msg)
    try:
        os.makedirs(folder_path)
    except:
        msg = f"Failed creating directory {folder_path}. Permissions?"
        raise LASIFCommandLineException(msg)

    Project(project_root_path=folder_path, init_project=folder_path.name)

    print(f"Initialized project in: \n\t{folder_path}")


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

    comm = _find_project_comm(".")
    comm.actions.finalize_adjoint_sources(iteration_name, event_name)


@mpi_enabled
@command_group("Iteration Management")
def lasif_calculate_all_adjoint_sources(parser, args):
    """
    Calculates all adjoint sources for a given iteration.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    args = parser.parse_args(args)

    iteration = args.iteration_name

    comm = _find_project_comm_mpi(".")

    events = comm.events.list()

    for _i, event in enumerate(events):
        if MPI.COMM_WORLD.rank == 0:
            print("\n{green}"
                  "==========================================================="
                  "{reset}".format(green=colorama.Fore.GREEN,
                                   reset=colorama.Style.RESET_ALL))
            print("Starting adjoint source calculation for event %i of "
                  "%i..." % (_i + 1, len(events)))
            print("{green}"
                  "==========================================================="
                  "{reset}\n".format(green=colorama.Fore.GREEN,
                                     reset=colorama.Style.RESET_ALL))

        # Windows file must exist!
        filename = comm.wins_and_adj_sources.get_filename(event=event,
                                                          iteration=iteration)
        if not os.path.exists(filename):
            if MPI.COMM_WORLD.rank == 0:
                print("File '%s' does not exists. No adjoint sources for "
                      "event '%s' will be calculated." % (filename, event))
            continue

        MPI.COMM_WORLD.barrier()
        comm.actions.calculate_adjoint_sources(event, iteration)


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
    parser.add_argument("event_name", help="name of the event")
    args = parser.parse_args(args)

    iteration_name = args.iteration
    event = args.event_name

    comm = _find_project_comm_mpi(".")

    comm.actions.select_windows(event, iteration_name)


@mpi_enabled
@command_group("Iteration Management")
def lasif_select_all_windows(parser, args):
    """
    Autoselect all windows for a given iteration.

    This function works with MPI. Don't use too many cores, I/O quickly
    becomes the limiting factor. It also works without MPI but then only one
    core actually does any work.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    args = parser.parse_args(args)

    iteration = args.iteration_name

    comm = _find_project_comm_mpi(".")

    events = comm.events.list()

    for _i, event in enumerate(events):
        if MPI.COMM_WORLD.rank == 0:
            print("\n{green}"
                  "==========================================================="
                  "{reset}".format(green=colorama.Fore.GREEN,
                                   reset=colorama.Style.RESET_ALL))
            print("Starting window selection for event %i of %i..." % (
                  _i + 1, len(events)))
            print("{green}"
                  "==========================================================="
                  "{reset}\n".format(green=colorama.Fore.GREEN,
                                     reset=colorama.Style.RESET_ALL))

        filename = comm.wins_and_adj_sources.get_filename(event=event,
                                                          iteration=iteration)
        if os.path.exists(filename):
            if MPI.COMM_WORLD.rank == 0:
                print("File '%s' already exists. Will not pick windows for that "
                      "event. Delete the file to repick windows." % filename)
            continue

        MPI.COMM_WORLD.barrier()
        comm.actions.select_windows(event, iteration)


@command_group("Iteration Management")
def lasif_launch_misfit_gui(parser, args):
    """
    Launch the misfit GUI.
    """
    args = parser.parse_args(args)

    comm = _find_project_comm(".")

    #TODO fix the ugly prints coming from this function
    from lasif.misfit_gui.misfit_gui import launch
    launch(comm)


@command_group("Iteration Management")
def lasif_create_weight_set(parser, args):
    """
    Create a new set of event and station weights.
    """
    parser.add_argument("weight_set_name", help="name of the weight set, i.e. \"A\"")

    args = parser.parse_args(args)
    weight_set_name = args.weight_set_name

    comm = _find_project_comm(".")
    comm.weights.create_new_weight_set(
        weight_set_name=weight_set_name,
        events_dict=comm.query.get_stations_for_all_events())

@command_group("Iteration Management")
def lasif_setup_new_iteration(parser, args):
    """
    Creates directory structure for a new iteration.
    """
    parser.add_argument("iteration_name", help="name of the iteration, i.e. \"1\"")

    args = parser.parse_args(args)
    iteration_name = args.iteration_name

    comm = _find_project_comm(".")
    comm.iterations.setup_directories_for_iteration(
        iteration_name=iteration_name)

@mpi_enabled
@command_group("Iteration Management")
def lasif_compare_misfits(parser, args):
    """
    Compares the misfit between two iterations. Will only consider windows
    that are identical in both iterations as the comparision is otherwise
    meaningless.
    """
    from lasif import LASIFAdjointSourceCalculationError

    parser.add_argument("from_iteration",
                        help="past iteration")
    parser.add_argument("to_iteration", help="current iteration")
    args = parser.parse_args(args)

    comm = _find_project_comm_mpi(".")

    _starting_time = time.time()

    # Read on each core as pickling/broadcasting them prooves to be
    # difficult. Should be possible if this ever becomes a performance issue.
    from_it = comm.iterations.get(args.from_iteration)
    to_it = comm.iterations.get(args.to_iteration)

    if MPI.COMM_WORLD.rank == 0:
        # Get a list of events that are in both, the new and the old iteration.
        events = sorted(set(from_it.events.keys()).intersection(
            set(to_it.events.keys())))
        event_count = len(events)

        # Split into a number of events per MPI process.
        events = split(events, MPI.COMM_WORLD.size)

        print(" => Calculating misfit change from iteration '%s' to " \
            "iteration '%s' ..." % (from_it.name, to_it.name))
        print(" => Launching calculations on %i core(s)\n" % \
            MPI.COMM_WORLD.size)

    else:
        events = None

    # Scatter jobs
    events = MPI.COMM_WORLD.scatter(events, root=0)

    total_misfit_from = 0
    total_misfit_to = 0

    all_events = collections.defaultdict(list)

    for _i, event in enumerate(events):
        # Get the windows from both.
        window_group_to = comm.windows.get(event, to_it)
        window_group_from = comm.windows.get(event, from_it)

        event_weight = from_it.events[event]["event_weight"]

        # Get a list of channels shared amongst both.
        shared_channels = set(window_group_to.list()).intersection(
            set(window_group_from.list()))

        # On rank 0, show a progressbar because it can take forever.
        if MPI.COMM_WORLD.rank == 0:
            widgets = [
                "Approximately event %i of %i: " % (
                    _i * MPI.COMM_WORLD.size + 1, event_count),
                progressbar.Percentage(),
                progressbar.Bar(), "", progressbar.ETA()]
            pbar = progressbar.ProgressBar(
                widgets=widgets, maxval=len(shared_channels)).start()

        for _i, channel in enumerate(shared_channels):
            if MPI.COMM_WORLD.rank == 0:
                pbar.update(_i)
            window_collection_from = window_group_from.get(channel)
            window_collection_to = window_group_to.get(channel)

            station_weight = from_it.events[event]["stations"][
                ".".join(channel.split(".")[:2])]["station_weight"]

            channel_misfit_from = 0
            channel_misfit_to = 0
            total_channel_weight = 0

            for win_from in window_collection_from.windows:
                try:
                    idx = window_collection_to.windows.index(win_from)
                    win_to = window_collection_to.windows[idx]
                except ValueError:
                    continue

                try:
                    misfit_from = win_from.misfit_value
                except LASIFAdjointSourceCalculationError:
                    continue
                except LASIFNotFoundError as e:
                    print(str(e))
                    continue

                try:
                    misfit_to = win_to.misfit_value
                except Exception as e:
                    print(e)
                    # Random penalty...but how else to compare?
                    misfit_to = 2.0 * misfit_from

                channel_misfit_from += misfit_from * win_from.weight
                channel_misfit_to += misfit_to * win_from.weight
                total_channel_weight += win_from.weight

            # Rare - but sometimes all windows for a certain channel fail
            # the calculation.
            if total_channel_weight == 0:
                continue

            # Make sure the misfits are consistent with the adjoint source
            # calculations!
            misfit_from *= event_weight * station_weight / total_channel_weight
            misfit_to *= event_weight * station_weight / total_channel_weight

            total_misfit_from += misfit_from
            total_misfit_to += misfit_to

            if (misfit_to - misfit_from) < -1.5:
                print(event, channel, misfit_from - misfit_to)
            all_events[event].append(misfit_to - misfit_from)
        if MPI.COMM_WORLD.rank == 0:
            pbar.finish()

    _all_events = MPI.COMM_WORLD.gather(all_events, root=0)

    total_misfit_from = MPI.COMM_WORLD.reduce(total_misfit_from, root=0)
    total_misfit_to = MPI.COMM_WORLD.reduce(total_misfit_to, root=0)

    # Only rank 0 continues.
    if MPI.COMM_WORLD.rank != 0:
        return

    # Collect in singular dictionary again.
    all_events = {}
    [all_events.update(_i) for _i in _all_events]

    if not all_events:
        raise LASIFCommandLineException("No misfit values could be compared.")

    print("\nTotal misfit in Iteration %s: %g" % (from_it.name,
                                                  total_misfit_from))
    print("Total misfit in Iteration %s: %g" % (to_it.name,
                                                total_misfit_to))

    _ending_time = time.time()

    print("\n => Computation time: %.1f seconds" % (_ending_time -
                                                    _starting_time))

    import matplotlib.pylab as plt
    import numpy as np

    plt.figure(figsize=(20, 3 * len(all_events)))
    plt.suptitle("Misfit change of measurements going from iteration"
                 " '%s' to iteration '%s'" % (from_it.name, to_it.name))
    for i, event_name in enumerate(sorted(all_events.keys())):
        values = np.array(all_events[event_name])
        colors = np.array(["green"] * len(values))
        colors[values > 0] = "red"
        plt.subplot(len(all_events), 1, i + 1)
        plt.bar(np.arange(len(values)), values, color=colors)
        plt.ylabel("difference")
        plt.xlim(0, len(values) - 1)
        plt.xticks([])
        plt.title("%i measurements with identical windows for event '%s'" %
                  (len(values), event_name))

    output_folder = comm.project.get_output_folder(
        type="misfit_comparisons", tag="misfit_comparision")
    filename = os.path.join(output_folder, "misfit_comparision.pdf")
    plt.savefig(filename)
    print("\nSaved figure to '%s'" % os.path.relpath(filename))


@command_group("Iteration Management")
def lasif_migrate_windows(parser, args):
    """
    Migrates windows from one iteration to the next.
    """
    parser.add_argument("from_iteration",
                        help="iteration containing windows")
    parser.add_argument("to_iteration", help="iteration windows will "
                                             "be migrated to")
    args = parser.parse_args(args)
    comm = _find_project_comm(".")

    from_it = comm.iterations.get(args.from_iteration)
    to_it = comm.iterations.get(args.to_iteration)

    print("Migrating windows from iteration '%s' to iteration '%s'..." % (
        from_it.name, to_it.name))

    for event_name, stations in to_it.events.items():
        stations = stations["stations"].keys()

        window_group_from = comm.windows.get(event_name, from_it.name)
        window_group_to = comm.windows.get(event_name, to_it.name)
        contents_to = set(window_group_to.list())
        contents_from = set(window_group_from.list())
        contents = contents_from - contents_to

        # Remove all not part of this iterations station.
        filtered_contents = itertools.ifilter(
            lambda x: ".".join(x.split(".")[:2]) in stations,
            contents)

        for channel_id in filtered_contents:
            coll = window_group_from.get(channel_id)
            coll.synthetics_tag = to_it.name
            f = window_group_to._get_window_filename(channel_id)
            coll.filename = f
            coll.write()


@command_group("Iteration Management")
def lasif_list_weight_sets(parser, args):
    """
    Print a list of all iterations in the project.
    """
    args = parser.parse_args(args)

    comm = _find_project_comm(".")

    it_len = comm.weights.count()

    print("%i weight set(s)%s in project:" % (it_len,
          "s" if it_len != 1 else ""))
    for weights in comm.weights.list():
        print("\t%s" % weights)


@command_group("Iteration Management")
def lasif_iteration_info(parser, args):
    """
    Print information about a single iteration.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name

    comm = _find_project_comm(".")
    if not comm.iterations.has_iteration(iteration_name):
        msg = ("Iteration '%s' not found. Use 'lasif list_iterations' to get "
               "a list of all available iterations.") % iteration_name
        raise LASIFCommandLineException(msg)

    print(comm.iterations.get(iteration_name))



@mpi_enabled
@command_group("Iteration Management")
def lasif_preprocess_data(parser, args):
    """
    Launch data preprocessing.

    This function works with MPI. Don't use too many cores, I/O quickly
    becomes the limiting factor. It also works without MPI but then only one
    core actually does any work.
    """
    parser.add_argument("weight_set", help="name of the weight set")
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")
    args = parser.parse_args(args)
    weight_set_name = args.weight_set
    events = args.events if args.events else None

    comm = _find_project_comm_mpi(".")

    # No need to perform these checks on all ranks.
    exceptions = []
    if MPI.COMM_WORLD.rank == 0:
        if not comm.weights.has_weight_set(weight_set_name):
           msg = ("Weights '%s' not found. Use 'lasif list_iterations' to "
                  "get a list of all available iterations.") % weight_set_name
           exceptions.append(msg)

        # Check if the event ids are valid.
        if not exceptions and events:
            for event_name in events:
                if not comm.events.has_event(event_name):
                    msg = "Event '%s' not found." % event_name
                    exceptions.append(msg)
                    break

    # Raise any exceptions on all ranks if necessary.
    exceptions = MPI.COMM_WORLD.bcast(exceptions, root=0)
    if exceptions:
        raise LASIFCommandLineException(exceptions[0])

    comm.actions.preprocess_data(weight_set_name, events)


@command_group("Plotting")
def lasif_plot_window_statistics(parser, args):
    """
    Plot the selected windows.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    args = parser.parse_args(args)

    iteration_name = args.iteration_name

    comm = _find_project_comm(".")

    if args.combine:
        comm.visualizations.plot_window_statistics(
            iteration=iteration_name, ax=None, show=True)


@command_group("Plotting")
def lasif_plot_windows(parser, args):
    """
    Plot the selected windows.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("event_name", help="name of the event")
    parser.add_argument(
        "--combine",
        help="Create a combined plot for all windows of that event.",
        action="store_true")
    parser.add_argument("--distance_bins", type=int,
                        help="The number of bins on the distance axis for "
                             "the combined plot.",
                        default=500)
    args = parser.parse_args(args)

    iteration_name = args.iteration_name
    event_name = args.event_name

    comm = _find_project_comm(".")

    if args.combine:
        comm.visualizations.plot_windows(event=event_name,
                                         iteration=iteration_name, ax=None,
                                         distance_bins=args.distance_bins,
                                         show=True)
    else:
        output_folder = comm.project.get_output_folder(
            type="plotted_windows",
            tag="Iteration_%s__%s" % (event_name, iteration_name))

        window_manager = comm.windows.get(event_name, iteration_name)
        for window_group in window_manager:
            window_group.plot(show=False, filename=os.path.join(output_folder,
                              "%s.png" % window_group.channel_id))
            sys.stdout.write(".")
            sys.stdout.flush()
        print("\nDone")

        print("Done. Written output to folder %s." % output_folder)


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

    comm = _find_project_comm(".")
    comm.validator.validate_data(
        station_file_availability=station_file_availability,
        raypaths=raypaths, waveforms=waveforms)


@command_group("Iteration Management")
def lasif_iteration_status(parser, args):
    """
    Query the current status of an iteration.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    args = parser.parse_args(args)
    iteration_name = args.iteration_name

    comm = _find_project_comm(".")
    status = comm.query.get_iteration_status(iteration_name)
    iteration = comm.iterations.get(iteration_name)

    print("Iteration %s is defined for %i events:" % (iteration_name,
                                                      len(iteration.events)))
    for event in sorted(status.keys()):
        st = status[event]
        print("\t%s" % event)

        print("\t\t%.2f %% of the events stations have picked windows" %
              (st["fraction_of_stations_that_have_windows"] * 100))
        if st["missing_raw"]:
            print("\t\tLacks raw data for %i stations" %
                  len(st["missing_raw"]))
        if st["missing_processed"]:
            print("\t\tLacks processed data for %i stations" %
                  len(st["missing_processed"]))
        if st["missing_synthetic"]:
            print("\t\tLacks synthetic data for %i stations" %
                  len(st["missing_synthetic"]))


def lasif_tutorial(parser, args):
    """
    Open the tutorial in a webbrowser.
    """
    parser.parse_args(args)

    import webbrowser
    webbrowser.open("http://krischer.github.io/LASIF/")


def lasif_debug(parser, args):
    """
    Print information LASIF can gather from a list of files.
    """
    parser.add_argument(
        "files", help="filenames to print debug information about", nargs="+")
    args = parser.parse_args(args)
    comm = _find_project_comm(".")

    for filename in args.files:
        filename = os.path.relpath(filename)
        if not os.path.exists(filename):
            print("{red}Path '{f}' does not exist.{reset}\n".format(
                f=filename, red=colorama.Fore.RED,
                reset=colorama.Style.RESET_ALL))
            continue
        print("{green}Path '{f}':{reset}".format(
            f=filename, green=colorama.Fore.GREEN,
            reset=colorama.Style.RESET_ALL))

        try:
            info = comm.query.what_is(filename)
        except LASIFError as e:
            info = "Error: %s" % e.message

        print("\t" + info)
        print("")


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
    parser.add_argument(
        "--open_to_outside",
        help="By default the website can only be opened from the current "
             "computer. Use this argument to access it from any other "
             "computer on the network.",
        action="store_true")
    args = parser.parse_args(args)
    port = args.port
    nobrowser = args.nobrowser
    debug = args.debug
    open_to_outside = args.open_to_outside

    if debug:
        nobrowser = True

    comm = _find_project_comm(".")

    if nobrowser is False:
        import webbrowser
        import threading

        threading.Timer(
            1.0, lambda: webbrowser.open("http://localhost:%i" % port)).start()

    from lasif.webinterface.server import serve
    serve(comm, port=port, debug=debug, open_to_outside=open_to_outside)


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
        prog="lasif %s" % fct.__name__.replace("lasif_", ""),
        description=_get_cmd_description(fct))

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
