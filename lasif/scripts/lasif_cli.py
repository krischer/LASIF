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
import colorama
import difflib
import pathlib
import sys
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
    parser.add_argument("--weight_set_name", help="for stations to be "
                                                  "color coded as a function "
                                                  "of their respective "
                                                  "weights", default=None)
    args = parser.parse_args(args)
    event_name = args.event_name

    comm = _find_project_comm(".")
    comm.visualizations.plot_event(event_name, args.weight_set_name)

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

    comm = _find_project_comm(".")
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
                             " Be very careful while using this one")
    args = parser.parse_args(args)
    providers = args.providers

    comm = _find_project_comm(".")
    event_name = args.event_name if args.event_name else comm.events.list()
    for event in event_name:
        comm.downloads.download_data(event, providers=providers)
        if args.downsample_data:
            # Do some kind of processesing
            print(f"Applying light processing to event: {event}")
            event_file = comm.waveforms.get_asdf_filename(event, "raw")
            print(f"Event file name: {event_file}")
            comm.waveforms.light_preprocess(event)
            print(f"Event: {event} successfully preprocessed")


@command_group("Event Management")
def lasif_list_events(parser, args):
    """
    Print a list of all events in the project.
    """
    parser.add_argument("--list", help="Show only a list of events. Good for "
                                       "scripting bash.",
                        action="store_true")
    args = parser.parse_args(args)

    from lasif.tools.prettytable import PrettyTable

    comm = _find_project_comm(".")

    if args.list is False:
        print("%i event%s in project:" % (comm.events.count(),
              "s" if comm.events.count() != 1 else ""))

    if args.list is True:
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


@command_group("Iteration Management")
def lasif_submit_all_jobs(parser, args):
    """
    EXPERIMENTAL: Submits all events to daint with salvus-flow. NO QA
    Requires all input_files to be present.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("ranks", help="amount of ranks", type=int)
    parser.add_argument("wall_time_in_seconds", help="wall time", type=int)
    parser.add_argument("simulation_type", help="forward, "
                                                "step_length, adjoint")
    import time

    args = parser.parse_args(args)

    iteration_name = args.iteration_name
    ranks = args.ranks
    wall_time = args.wall_time_in_seconds
    simulation_type = args.simulation_type
    comm = _find_project_comm(".")

    long_iter_name = comm.iterations.get_long_iteration_name(
        iteration_name)
    input_files_dir = comm.project.paths['salvus_input']

    events = comm.events.list()
    for event in events:
        file = os.path.join(input_files_dir, long_iter_name, event,
                            simulation_type, "run_salvus.sh")
        job_name = f"{event}_{long_iter_name}_{simulation_type}"
        command = f"salvus-flow run-salvus --site daint " \
                  f"--wall-time-in-seconds {wall_time} " \
                  f"--custom-job-name {job_name} " \
                  f"--ranks {ranks} {file}"

        if simulation_type == "adjoint":
            command += f" --wavefield-job-name" \
                       f" {event}_{long_iter_name}_forward@daint"
        os.system(command)
        time.sleep(2)


@command_group("Iteration Management")
def lasif_retrieve_all_output(parser, args):
    """
    EXPERIMENTAL: Retrieves all output from daint, No QA.
    """
    parser.add_argument("iteration_name", help="name of the iteration")
    parser.add_argument("simulation_type", help="forward, "
                                                "step_length, adjoint")

    args = parser.parse_args(args)
    comm = _find_project_comm(".")
    import time
    iteration_name = args.iteration_name
    simulation_type = args.simulation_type

    long_iter_name = comm.iterations.get_long_iteration_name(
        iteration_name)

    if simulation_type in ["forward", "step_length"]:
        base_dir = comm.project.paths["eq_synthetics"]
    else:
        base_dir = comm.project.paths["gradients"]
    events = comm.events.list()
    import shutil
    for event in events:
        output_dir = os.path.join(base_dir, long_iter_name, event)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        job_name = f"{event}_{long_iter_name}_{simulation_type}@daint"
        command = f"salvus-flow get-output {job_name} {output_dir}"
        os.system(command)
        time.sleep(2)
        print(f"Retrieved {job_name}")


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

    freqmax = 1.0 / comm.project.processing_params["highpass_period"]
    freqmin = 1.0 / comm.project.processing_params["lowpass_period"]

    stf_fct = comm.project.get_project_function(
        "source_time_function")

    delta = comm.project.solver_settings["time_increment"]
    npts = comm.project.solver_settings["number_of_time_steps"]

    stf = {"delta": delta}

    stf["data"] = stf_fct(npts=npts, delta=delta,
                          freqmin=freqmin, freqmax=freqmax)

    # Ignore lots of potential warnings with some plotting functionality.
    lasif.visualization.plot_tf(stf["data"], stf["delta"], freqmin=freqmin,
                                freqmax=freqmax)


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

    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    weight_set_name = args.weight_set_name
    simulation_type = args.simulation_type

    simulation_type_options = ["forward", "step_length", "adjoint"]
    if simulation_type not in simulation_type_options:
        raise LASIFError("Please choose simulation_type from: "
                         "[%s]" % ", ".join(map(str, simulation_type_options)))

    comm = _find_project_comm(".")
    events = args.events if args.events else comm.events.list()

    if weight_set_name:
        if not comm.weights.has_weight_set(weight_set_name):
            raise LASIFNotFoundError(f"Weights {weight_set_name} not known"
                                     f"to LASIF")

    if not comm.iterations.has_iteration(iteration_name):
        raise LASIFNotFoundError(f"Could not find iteration: {iteration_name}")

    for _i, event in enumerate(events):
        if not comm.events.has_event(event):
                print(f"Event {event} not known to LASIF. "
                      f"No input files for this event"
                      f" will be generated. ")
        print(f"Generating input files for event "
              f"{_i + 1} of {len(events)} -- {event}")
        if simulation_type == "adjoint":
            comm.actions.finalize_adjoint_sources(iteration_name, event,
                                                  weight_set_name)
        else:
            comm.actions.generate_input_files(iteration_name, event,
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
    iteration = args.iteration_name
    window_set_name = args.window_set_name
    comm = _find_project_comm_mpi(".")

    # some basic checks
    if not comm.windows.has_window_set(window_set_name):
        if MPI.COMM_WORLD.rank == 0:
            raise LASIFNotFoundError(
                "Window set {} not known to LASIF".format(window_set_name))
        return

    if not comm.iterations.has_iteration(iteration):
        if MPI.COMM_WORLD.rank == 0:
            raise LASIFNotFoundError(
                "Iteration {} not known to LASIF".format(iteration))
        return

    events = args.events if args.events else comm.events.list()

    for _i, event in enumerate(events):
        if not comm.events.has_event(event):
            if MPI.COMM_WORLD.rank == 0:
                print("Event '%s' not known to LASIF. No adjoint sources for "
                      "this event will be calculated. " % event)
            continue

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

        # Get adjoint sources_filename
        filename = comm.adj_sources.get_filename(event=event,
                                                 iteration=iteration)
        # remove adjoint sources if they already exist
        if MPI.COMM_WORLD.rank == 0:
            if os.path.exists(filename):
                os.remove(filename)

        MPI.COMM_WORLD.barrier()
        comm.actions.calculate_adjoint_sources(event, iteration,
                                               window_set_name)


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

    comm = _find_project_comm_mpi(".")

    iteration_name = args.iteration
    window_set_name = args.window_set_name
    events = args.events if args.events else comm.events.list()
    for event in events:
        print(f"Selecting windows for event: {event}")
        comm.actions.select_windows(event, iteration_name, window_set_name)


@command_group("Iteration Management")
def lasif_launch_misfit_gui(parser, args):
    """
    Launch the misfit GUI.
    """
    args = parser.parse_args(args)
    comm = _find_project_comm(".")

    from lasif.misfit_gui.misfit_gui import launch
    launch(comm)


@command_group("Iteration Management")
def lasif_create_weight_set(parser, args):
    """
    Create a new set of event and station weights.
    """
    parser.add_argument("weight_set_name",
                        help="name of the weight set, i.e. \"A\"")

    args = parser.parse_args(args)
    weight_set_name = args.weight_set_name

    comm = _find_project_comm(".")
    comm.weights.create_new_weight_set(
        weight_set_name=weight_set_name,
        events_dict=comm.query.get_stations_for_all_events())


@command_group("Iteration Management")
def lasif_compute_station_weights(parser, args):
    """
    Compute weights for stations. Useful when distribution is uneven.
    Weights are calculated for each event. This may take a while if you have
    many stations.
    """
    import progressbar
    parser.add_argument("weight_set_name", help="Pick a name for the "
                                                "weight set. If the weight"
                                                "set already exists, it will"
                                                "overwrite")
    parser.add_argument("event_name", default=None,
                        help="name of event. If none is specified weights will"
                             "be calculated for all. Also possible to specify"
                             "more than one separated by a space", nargs="*")
    args = parser.parse_args(args)
    w_set = args.weight_set_name

    comm = _find_project_comm(".")

    event_name = args.event_name if args.event_name else comm.events.list()

    if not comm.weights.has_weight_set(w_set):
        print("Weight set does not exist. Will create new one.")
        comm.weights.create_new_weight_set(
            weight_set_name=w_set,
            events_dict=comm.query.get_stations_for_all_events())

    weight_set = comm.weights.get(w_set)
    s = 0
    bar = progressbar.ProgressBar(max_value=len(event_name))
    for event in event_name[:1]:
        if not comm.events.has_event(event):
            raise LASIFNotFoundError(f"Event: {event} is not known to LASIF")
        stations = comm.query.get_all_stations_for_event(event)
        sum_value = 0.0

        for station in stations:
            weight = comm.weights.calculate_station_weight(station, stations)
            sum_value += weight
            weight_set.events[event]["stations"][station]["station_weight"] = \
                weight
        s += 1
        bar.update(s)
        for station in stations:
            weight_set.events[event]["stations"][station]["station_weight"] *=\
                (len(stations) / sum_value)

    comm.weights.change_weight_set(
        weight_set_name=w_set, weight_set=weight_set,
        events_dict=comm.query.get_stations_for_all_events())


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

    args = parser.parse_args(args)
    iteration_name = args.iteration_name
    remove_dirs = args.remove_dirs

    comm = _find_project_comm(".")
    comm.iterations.setup_directories_for_iteration(
        iteration_name=iteration_name, remove_dirs=remove_dirs)


@command_group("Iteration Management")
def lasif_list_iterations(parser, args):
    """
    Creates directory structure for a new iteration.
    """
    args = parser.parse_args(args)

    comm = _find_project_comm(".")
    iterations = comm.iterations.list()
    print("Iterations known to LASIF: \n")
    for iteration in iterations:
        print(comm.iterations.get_long_iteration_name(iteration), "\n")


@mpi_enabled
@command_group("Iteration Management")
def lasif_compare_misfits(parser, args):
    """
    Compares the total misfit between two iterations. Total misfit
    is used regardless of the similarity of the picked windows
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
    parser.add_argument("--weight_set_name", default=None, type=str,
                        help="Set of station and event weights")
    parser.add_argument("--print_events", help="compare misfits"
                                               " for each event",
                        action="store_true")
    args = parser.parse_args(args)

    comm = _find_project_comm_mpi(".")

    from_it = args.from_iteration
    to_it = args.to_iteration
    weight_set_name = args.weight_set_name

    if weight_set_name:
        if not comm.weights.has_weight_set(weight_set_name):
            raise LASIFNotFoundError(f"Weights {weight_set_name} not known"
                                     f"to LASIF")
    # Check if iterations exist
    if not comm.iterations.has_iteration(from_it):
            raise LASIFNotFoundError(f"Iteration {from_it} not known to LASIF")
    if not comm.iterations.has_iteration(to_it):
            raise LASIFNotFoundError(f"Iteration {to_it} not known to LASIF")

    from_it_misfit = 0.0
    to_it_misfit = 0.0
    for event in comm.events.list():
        from_it_misfit += \
            comm.adj_sources.get_misfit_for_event(event,
                                                  args.from_iteration,
                                                  weight_set_name)
        to_it_misfit += \
            comm.adj_sources.get_misfit_for_event(event,
                                                  args.to_iteration,
                                                  weight_set_name)
        if args.print_events:
            # Print information about every event.
            from_it_misfit_event = \
                comm.adj_sources.get_misfit_for_event(event,
                                                      args.from_iteration,
                                                      weight_set_name)
            to_it_misfit_event = \
                comm.adj_sources.get_misfit_for_event(event,
                                                      args.to_iteration,
                                                      weight_set_name)
            print(f"{event}: \n"
                  f"\t iteration {from_it} has misfit: "
                  f"{from_it_misfit_event} \n"
                  f"\t iteration {to_it} has misfit: {to_it_misfit_event}.")

    print(f"Total misfit for iteration {from_it}: {from_it_misfit}")
    print(f"Total misfit for iteration {to_it}: {to_it_misfit}")
    rel_change = (to_it_misfit - from_it_misfit) / from_it_misfit
    print(f"Relative change in total misfit from iteration {from_it} to "
          f"{to_it} is: {rel_change}")
    n_events = len(comm.events.list())
    print(f"Misfit per event for iteration {from_it}: "
          f"{from_it_misfit/n_events}")
    print(f"Misfit per event for iteration {to_it}: "
          f"{to_it_misfit/n_events}")


@command_group("Iteration Management")
def lasif_list_weight_sets(parser, args):
    """
    Print a list of all iterations in the project.
    """
    comm = _find_project_comm(".")
    it_len = comm.weights.count()

    print("%i weight set(s)%s in project:" % (it_len,
          "s" if it_len != 1 else ""))
    for weights in comm.weights.list():
        print("\t%s" % weights)


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
    args = parser.parse_args(args)
    comm = _find_project_comm_mpi(".")
    events = args.events if args.events else comm.events.list()

    # No need to perform these checks on all ranks.
    exceptions = []
    if MPI.COMM_WORLD.rank == 0:
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
    comm.actions.process_data(events)


@command_group("Plotting")
def lasif_plot_window_statistics(parser, args):
    """
    Plot the selected windows.
    """
    parser.add_argument("window_set_name", help="name of the window set")
    parser.add_argument(
        "events", help="One or more events. If none given, all will be done.",
        nargs="*")
    args = parser.parse_args(args)

    window_set_name = args.window_set_name
    comm = _find_project_comm(".")

    if not comm.windows.has_window_set(window_set_name):
        raise LASIFNotFoundError("Could not find the specified window set")

    events = args.events if args.events else comm.events.list()
    comm.visualizations.plot_window_statistics(
        window_set_name, events, ax=None, show=True)


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

    event_name = args.event_name
    window_set_name = args.window_set_name
    comm = _find_project_comm(".")

    comm.visualizations.plot_windows(event=event_name,
                                     window_set_name=window_set_name, ax=None,
                                     distance_bins=args.distance_bins,
                                     show=True)


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
    full_check = args.full
    data_and_station_file_availability = \
        args.data_and_station_file_availability
    raypaths = args.raypaths

    # If full check, check everything.
    if full_check:
        data_and_station_file_availability = True
        raypaths = True

    comm = _find_project_comm(".")
    comm.validator.validate_data(
        data_and_station_file_availability=data_and_station_file_availability,
        raypaths=raypaths)


def lasif_tutorial(parser, args):
    """
    Open the tutorial in a webbrowser.
    """
    parser.parse_args(args)

    import webbrowser
    webbrowser.open("http://dirkphilip.github.io/LASIF_2.0/")


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
                  colorama.Fore.CYAN,
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
