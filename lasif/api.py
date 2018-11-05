#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An api to communicate directly with lasif functions without using the
command line interface.

:copyright:
    Solvi Thrastarson (soelvi.thrastarson@erdw.ethz.ch), 2018
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

"""
import os
import pathlib

import colorama
from mpi4py import MPI
import toml

from lasif import LASIFError
from lasif.components.communicator import Communicator
from lasif.components.project import Project
from lasif import LASIFNotFoundError


class LASIFCommandLineException(Exception):
    pass


def find_project_comm(folder):
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


def plot_domain(lasif_root, save, show_mesh=False):
    """
    Plot the studied domain specified in config file
    :param lasif_root: path to lasif root directory.
    :param save: save file
    """
    import matplotlib.pyplot as plt
    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_domain(show_mesh=show_mesh)

    if save:
        outfile = os.path.join(comm.project.get_output_folder(
            type="domain_plot", tag="domain", timestamp=True), "domain.png")
        plt.savefig(outfile, dpi=200, transparent=True)
        print(f"Saved picture at {outfile}")
    else:
        plt.show()


def plot_event(lasif_root, event_name, weight_set_name, save, show_mesh=False):
    """
    Plot a single event including stations on a map. Events can be
    color coded based on their weight
    :param lasif_root: path to lasif root directory
    :param event_name: name of event to plot
    :param weight_set_name: name of station weight set
    :param save: if figure should be saved
    """
    import matplotlib.pyplot as plt

    comm = find_project_comm(lasif_root)

    if save:
        plt.switch_backend('agg')

    comm.visualizations.plot_event(event_name, weight_set_name,
                                   show_mesh=show_mesh)

    if save:
        outfile = os.path.join(
            comm.project.get_output_folder(
                type="event_plots", tag="event", timestamp=False),
            f"{event_name}.png", )
        plt.savefig(outfile, dpi=200, transparent=True)
        print("Saved picture at %s" % outfile)
    else:
        plt.show()


def plot_events(lasif_root, type, iteration, save, show_mesh=False):
    """
    Plot a all events on the domain
    :param lasif_root: path to lasif root directory
    :param type: type of plot
    :param iteration: plot all events of an iteration
    :param save: if figure should be saved
    """
    import matplotlib.pyplot as plt

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_events(type, iteration=iteration,
                                    show_mesh=show_mesh)

    if save:
        if iteration:
            file = f"events_{iteration}.png"
        else:
            file = "events.png"
        outfile = os.path.join(
            comm.project.get_output_folder(
                type="event_plots", tag="events", timestamp=True),
            file)
        plt.savefig(outfile, dpi=200, transparent=True)
        print("Saved picture at %s" % outfile)
    else:
        plt.show()


# @TODO: Add an option of plotting for a specific iteration
# @TODO: Make sure coastlines are plotted
def plot_raydensity(lasif_root, plot_stations):
    """
    Plot a distribution of earthquakes and stations with great circle rays
    plotted underneath.
    :param lasif_root: Lasif root directory
    :param plot_stations: boolean argument whether stations should be plotted
    """

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_raydensity(plot_stations=plot_stations)


def add_gcmt_events(lasif_root, count, min_mag, max_mag, min_dist,
                    min_year=None, max_year=None):
    """
    Add events to the project.
    :param lasif_root: path to lasif root directory
    :param count: Amount of events
    :param min_mag: Minimum magnitude
    :param max_mag: Maximum magnitude
    :param min_dist: Minimum distance between events
    :param min_year: Year to start event search [optional]
    :param max_year: Year to end event search [optional]
    """

    from lasif.tools.query_gcmt_catalog import add_new_events

    comm = find_project_comm(lasif_root)

    add_new_events(comm=comm, count=count, min_magnitude=min_mag,
                   max_magnitude=max_mag, min_year=min_year, max_year=max_year,
                   threshold_distance_in_km=min_dist)


def project_info(lasif_root):
    """
    Print a summary of the project
    :param lasif_root: path to lasif root directory
    """

    comm = find_project_comm(lasif_root)
    print(comm.project)


def download_data(lasif_root, event_name=[], providers=None):
    """
    Download available data for events
    :param lasif_root: path to lasif root directory
    :param event_name: Name of event [optional]
    :param providers: Providers to download from [optional]
    """

    comm = find_project_comm(lasif_root)
    if len(event_name) == 0:
        event_name = comm.events.list()

    for event in event_name:
        comm.downloads.download_data(event, providers=providers)


def list_events(lasif_root, list=True, iteration=None):
    """
    Print a list of events in project
    :param lasif_root: path to lasif root directory
    :param list: Show only a list of events, good for scripting [optional]
    :param iteration: Show only events for specific iteration [optional]
    """

    comm = find_project_comm(lasif_root)

    if list:
        for event in sorted(comm.events.list(iteration=iteration)):
            print(event)

    else:
        print(f"%i event%s in %s:" % (
            comm.events.count(), "s" if comm.events.count() != 1 else "",
            "iteration" if iteration else "project"))

        from lasif.tools.prettytable import PrettyTable

        tab = PrettyTable(["Event Name", "Lat/Lng/Depth(km)/Mag"])
        tab.align["Event Name"] = "l"
        for event in comm.events.list(iteration=iteration):
            ev = comm.events.get(event)
            tab.add_row([
                event, "%6.1f / %6.1f / %3i / %3.1f" % (
                    ev["latitude"], ev["longitude"], int(ev["depth_in_km"]),
                    ev["magnitude"])])
        print(tab)


def submit_job(lasif_root, iteration, ranks, wall_time, simulation_type, site,
               events=[]):
    """
    Submits job(s) to a supercomputer using salvus-flow. Be careful with this
    one
    :param lasif_root: path to lasif root directory
    :param iteration: Name of iteration
    :param ranks: Amount of CPUs to use
    :param wall_time: Wall time in seconds
    :param simulation_type: forward, adjoint, step_length
    :param site: Which computer to send job to
    :param events: If you only want to submit selected events. [optional]
    """
    import salvus_flow.api

    if isinstance(lasif_root, Communicator):
        comm = lasif_root
    else:
        comm = find_project_comm(lasif_root)

    if simulation_type not in ["forward", "step_length", "adjoint"]:
        raise LASIFError("Simulation type needs to be forward, step_length"
                         " or adjoint")

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

    long_iter_name = comm.iterations.get_long_iteration_name(iteration)
    input_files_dir = comm.project.paths["salvus_input"]

    for event in events:
        file = os.path.join(input_files_dir, long_iter_name, event,
                            simulation_type, "run_salvus.sh")
        job_name = f"{event}_{long_iter_name}_{simulation_type}"

        if simulation_type == "adjoint":
            wave_job_name = f"{event}_{long_iter_name}_forward@{site}"
            salvus_flow.api.run_salvus(site=site, cmd_line=file,
                                       wall_time_in_seconds=wall_time,
                                       custom_job_name=job_name, ranks=ranks,
                                       wavefield_job_name=wave_job_name)
        else:
            salvus_flow.api.run_salvus(site=site, cmd_line=file,
                                       wall_time_in_seconds=wall_time,
                                       custom_job_name=job_name, ranks=ranks)


def retrieve_output(lasif_root, iteration, simulation_type, site, events=[]):
    """
    Retrieve output from simulation
    :param lasif_root: path to lasif root directory
    :param iteration: Name of iteration
    :param simulation_type: forward, adjoint, step_length
    :param site: Which computer to send job to
    :param events: If you only want to submit selected events. [optional]
    """

    import salvus_flow.api
    import shutil

    if isinstance(lasif_root, Communicator):
        comm = lasif_root
    else:
        comm = find_project_comm(lasif_root)

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

    long_iter_name = comm.iterations.get_long_iteration_name(iteration)

    if simulation_type in ["forward", "step_length"]:
        base_dir = comm.project.paths["eq_synthetics"]
    elif simulation_type == "adjoint":
        base_dir = comm.project.paths["gradients"]
    else:
        raise LASIFError("Simulation type needs to be forward, step_length"
                         " or adjoint")

    for event in events:
        output_dir = os.path.join(base_dir, long_iter_name, event)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        job_name = f"{event}_{long_iter_name}_{simulation_type}@{site}"
        salvus_flow.api.get_output(job=job_name, destination=output_dir)


def event_info(lasif_root, event_name, verbose):
    """
    Print information about a single event
    :param lasif_root: path to lasif root directory
    :param event_name: Name of event
    :param verbose: Print station information as well
    """

    comm = find_project_comm(lasif_root)
    if not comm.events.has_event(event_name):
        msg = "Event '%s' not found in project." % event_name
        raise LASIFError(msg)

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


def plot_stf(lasif_root):
    """
    Plot the source time function
    :param lasif_root: path to lasif root directory
    """
    import lasif.visualization

    comm = find_project_comm(lasif_root)

    freqmax = 1.0 / comm.project.processing_params["highpass_period"]
    freqmin = 1.0 / comm.project.processing_params["lowpass_period"]

    stf_fct = comm.project.get_project_function(
        "source_time_function")

    delta = comm.project.solver_settings["time_increment"]
    npts = comm.project.solver_settings["number_of_time_steps"]
    stf_type = comm.project.solver_settings["source_time_function_type"]

    stf = {"delta": delta}
    if stf_type == "heaviside":
        stf["data"] = stf_fct(npts=npts, delta=delta)
        lasif.visualization.plot_heaviside(stf["data"], stf["delta"])
    elif stf_type == "bandpass_filtered_heaviside":
        stf["data"] = stf_fct(npts=npts, delta=delta, freqmin=freqmin,
                              freqmax=freqmax)
        lasif.visualization.plot_tf(stf["data"], stf["delta"], freqmin=freqmin,
                                    freqmax=freqmax)
    else:
        raise LASIFError(f"{stf_type} is not supported by lasif. Check your"
                         f"config file and make sure the source time "
                         f"function is either 'heaviside' or "
                         f"'bandpass_filtered_heaviside'.")


def generate_input_files(lasif_root, iteration, simulation_type, events=[],
                         weight_set=None, prev_iter=None):
    """
    Generate input files to prepare for numerical simulations
    :param lasif_root: path to lasif root directory
    :param iteration: name of iteration
    :param simulation_type: forward, step_length, adjoint
    :param events: One or more events [optional]
    :param weight_set: Name of weight set [optional]
    :param prev_iter: If input files can be copied from another iteration
    [optional]
    """

    comm = find_project_comm(lasif_root)

    simulation_type_options = ["forward", "step_length", "adjoint"]
    if simulation_type not in simulation_type_options:
        raise LASIFError("Please choose simulation_type from: "
                         "[%s]" % ", ".join(map(str, simulation_type_options)))

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

    if weight_set:
        if not comm.weights.has_weight_set(weight_set):
            raise LASIFNotFoundError(f"Weights {weight_set} not known"
                                     f"to LASIF")

    if not comm.iterations.has_iteration(iteration):
        raise LASIFNotFoundError(f"Could not find iteration: {iteration}")

    if prev_iter and \
            not comm.iterations.has_iteration(prev_iter):
        raise LASIFNotFoundError(f"Could not find iteration:"
                                 f" {prev_iter}")

    for _i, event in enumerate(events):
        if not comm.events.has_event(event):
                print(f"Event {event} not known to LASIF. "
                      f"No input files for this event"
                      f" will be generated. ")
                continue
        print(f"Generating input files for event "
              f"{_i + 1} of {len(events)} -- {event}")
        if simulation_type == "adjoint":
            comm.actions.finalize_adjoint_sources(iteration, event,
                                                  weight_set)

        else:
            comm.actions.generate_input_files(iteration, event,
                                              simulation_type, prev_iter)

    comm.iterations.write_info_toml(iteration, simulation_type)


def init_project(project_path):
    """
    Create a new project
    :param project_path: Path to project root directory. Can use absolute
    paths or relative paths from current working directory.
    """

    project_path = pathlib.Path(project_path).absolute()

    if project_path.exists():
        msg = "The given PROJECT_PATH already exists. It must not exist yet."
        raise LASIFError(msg)
    try:
        os.makedirs(project_path)
    except Exception:
        msg = f"Failed creating directory {project_path}. Permissions?"
        raise LASIFError(msg)

    Project(project_root_path=project_path, init_project=project_path.name)


def calculate_adjoint_sources(lasif_root, iteration, window_set,
                              events=[]):
    """
    Calculate adjoint sources for a given iteration
    :param lasif_root: path to lasif root directory
    :param iteration: name of iteration
    :param window_set: name of window set
    :param events: events [optional]
    """

    comm = find_project_comm(lasif_root)

    # some basic checks
    if not comm.windows.has_window_set(window_set):
        if MPI.COMM_WORLD.rank == 0:
            raise LASIFNotFoundError(
                "Window set {} not known to LASIF".format(window_set))
        return

    if not comm.iterations.has_iteration(iteration):
        if MPI.COMM_WORLD.rank == 0:
            raise LASIFNotFoundError(
                "Iteration {} not known to LASIF".format(iteration))
        return

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

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
                                               window_set)


def select_windows(lasif_root, iteration, window_set, events=[]):
    """
    Autoselect windows for a given iteration and event combination
    :param lasif_root: path to lasif root directory
    :param iteration: name of iteration
    :param window_set: name of window set
    :param events: events [optional]
    """

    comm = find_project_comm(lasif_root)  # Might have to do this mpi

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

    for event in events:
        print(f"Selecting windows for event: {event}")
        comm.actions.select_windows(event, iteration, window_set)


def open_gui(lasif_root):
    """
    Open up the misfit gui
    :param lasif_root: path to lasif root directory
    """

    comm = find_project_comm(lasif_root)

    from lasif.misfit_gui.misfit_gui import launch
    launch(comm)


def create_weight_set(lasif_root, weight_set):
    """
    Create a new set of event and station weights
    :param lasif_root: path to lasif root directory
    :param weight_set: name of weight set
    """

    comm = find_project_comm(lasif_root)

    comm.weights.create_new_weight_set(weight_set_name=weight_set,
                                       events_dict=comm.query.
                                       get_all_stations_for_events())


def compute_station_weights(lasif_root, weight_set, events=[], iteration=None):
    """
    Compute weights for stations based on amount of neighbouring stations
    :param lasif_root: path to lasif root directory
    :param weight_set: name of weight set to compute into
    :param events: name of event [optional]
    :param iteration: name of iteration to compute weights for [optional]
    """
    import progressbar
    comm = find_project_comm(lasif_root)

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

    if not comm.weights.has_weight_set(weight_set):
        print("Weight set does not exist. Will create new one.")
        comm.weights.create_new_weight_set(
            weight_set_name=weight_set,
            events_dict=comm.query.get_stations_for_all_events())

    w_set = comm.weights.get(weight_set)
    s = 0

    if len(events) == 1:
        one_event = True
    else:
        one_event = False
        bar = progressbar.ProgressBar(max_value=len(events))

    for event in events:
        if not comm.events.has_event(event):
            raise LASIFNotFoundError(f"Event: {event} is not known to LASIF")
        stations = comm.query.get_all_stations_for_event(event)
        sum_value = 0.0

        for station in stations:
            weight = comm.weights.calculate_station_weight(station, stations)
            sum_value += weight
            w_set.events[event]["stations"][station]["station_weight"] = \
                weight
        for station in stations:
            w_set.events[event]["stations"][station]["station_weight"] *= \
                (len(stations) / sum_value)
        if not one_event:
            s += 1
            bar.update(s)

    comm.weights.change_weight_set(
        weight_set_name=weight_set, weight_set=w_set,
        events_dict=comm.query.get_stations_for_all_events())


def get_weighting_bins(lasif_root, window_set, events=[], iteration=None):
    """
    Compute average envelopes for the observed data in certain station bins.
    The binning is based on event-station distances.
    :param lasif_root: path to lasif root directory
    :param window_set: name of window set, station with no windows not used.
    :param events: events to include [optional]
    :param iteration: use all events for an iteration [optional]
    """

    from obspy.geodetics.base import gps2dist_azimuth, kilometer2degrees
    import obspy.signal.filter
    import operator
    from collections import OrderedDict
    import pyasdf
    import numpy as np
    comm = find_project_comm(lasif_root)

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

    windows = comm.windows.get(window_set)

    distances = OrderedDict()

    # Collect information on event-station distances.
    # Could be put into a new function
    for event_name in events:
        distances[event_name] = OrderedDict()
        stations = comm.query.get_all_stations_for_event(event_name)
        event = comm.events.get(event_name)
        for station_id in stations:
            if len(windows.get_all_windows_for_event_station(
                    event_name, station_id)) == 0:
                continue
            station = comm.query.get_coordinates_for_station(
                event_name, station_id)
            dist = gps2dist_azimuth(event["latitude"], event["longitude"],
                                    station["latitude"], station["longitude"])
            dist = kilometer2degrees(dist[0] / 1000.0)
            distances[event_name][station_id] = dist

    # Sort the stations using distances. Bin them appropriately
    bins_dict = OrderedDict()
    for event_name in events:
        asdf_file = comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="processed",
            tag_or_iteration=comm.waveforms.preprocessing_tag)
        with pyasdf.ASDFDataSet(asdf_file) as ds:
            if "BinEnvelope" in ds.auxiliary_data.list():
                del ds.auxiliary_data["BinEnvelope"]
            sorted_stations = sorted(distances[event_name].items(),
                                     key=operator.itemgetter(1))
            dist = [x[1]for x in sorted_stations]

            # If stop=max(dist) we will have a bin with one station
            stop = min(dist) + (dist[-1] - dist[0]) * 0.9
            bins = np.linspace(start=min(dist), stop=stop, num=10)
            # Organize the stations into bins
            bin_array = np.digitize(
                x=np.array(list(distances[event_name].values())), bins=bins)
            i = 0
            bins_dict[event_name] = OrderedDict()
            for station in distances[event_name]:
                bins_dict[event_name][station] = bin_array[i]
                i += 1

            # Now we need to get waveforms and calculate envelopes.
            for s in range(1, len(bins) + 1):
                stations = [k for k, v in
                            bins_dict[event_name].items() if v == s]
                storage = np.zeros((len(stations),
                                    len(ds.waveforms[ds.waveforms.list()[0]]
                                        [comm.waveforms.preprocessing_tag]
                                        [0].data)))
                i = 0
                for station in stations:
                    print(station)
                    wave = comm.waveforms.get_waveforms_processed(
                        event_name=event_name, station_id=station,
                        tag=comm.waveforms.preprocessing_tag)
                    wave_z = wave.select(component="Z")
                    data_envelope = obspy.signal.filter.envelope(
                        wave_z[0].data)

                    storage[i, :] = data_envelope
                    i += 1
                if len(stations) == 0:
                    envelope = np.zeros(len(data_envelope))
                else:
                    envelope = np.mean(storage, axis=0)

                if s == len(bins):
                    parameters = {"bin": s,
                                  "min_dist": bins[s - 1],
                                  "max_dist": "INFINITE",
                                  "num_stations": len(stations),
                                  "stations": ", ".join(stations)}
                else:
                    parameters = {"bin": s,
                                  "min_dist": bins[s - 1],
                                  "max_dist": bins[s],
                                  "num_stations": len(stations),
                                  "stations": ", ".join(stations)}
                data_type = "BinEnvelope"
                path = f"{s}_bin"
                ds.add_auxiliary_data(data=envelope, data_type=data_type,
                                      path=path, parameters=parameters)


def set_up_iteration(lasif_root, iteration, events=[], remove_dirs=False):
    """
    Creates or removes directory structure for an iteration
    :param lasif_root: path to lasif root directory
    :param iteration: name of iteration
    :param events: events to include in iteration [optional]
    :param remove_dirs: boolean value to remove dirs [default=False]
    """

    comm = find_project_comm(lasif_root)

    if len(events) == 0:
        events = comm.events.list()

    comm.iterations.setup_directories_for_iteration(
        iteration_name=iteration, remove_dirs=remove_dirs)

    if not remove_dirs:
        comm.iterations.setup_iteration_toml(iteration_name=iteration)
        comm.iterations.setup_events_toml(iteration_name=iteration,
                                          events=events)


def write_misfit(lasif_root, iteration, weight_set=None, window_set=None):
    """
    Write misfit for iteration
    :param lasif_root: path to lasif root directory
    :param iteration: name of iteration
    :param weight_set: name of weight set [optional]
    :param window_set: name of window set [optional]
    """

    comm = find_project_comm(lasif_root)

    if weight_set:
        if not comm.weights.has_weight_set(weight_set):
            raise LASIFNotFoundError(f"Weights {weight_set} not known"
                                     f"to LASIF")
    # Check if iterations exist
    if not comm.iterations.has_iteration(iteration):
            raise LASIFNotFoundError(f"Iteration {iteration} "
                                     f"not known to LASIF")

    events = comm.events.list(iteration=iteration)

    total_misfit = 0.0
    iteration_dict = {"event_misfits": {}}
    for event in events:
        event_misfit = \
            comm.adj_sources.get_misfit_for_event(event,
                                                  iteration,
                                                  weight_set)
        iteration_dict["event_misfits"][event] = float(event_misfit)
        total_misfit += event_misfit

    iteration_dict["total_misfit"] = total_misfit
    iteration_dict["weight_set_name"] = weight_set
    iteration_dict["window_set_name"] = window_set

    long_iter_name = comm.iterations.get_long_iteration_name(iteration)

    path = comm.project.paths["iterations"]
    toml_filename = os.path.join(path, long_iter_name, "misfits.toml")

    with open(toml_filename, "w") as fh:
        toml.dump(iteration_dict, fh)


def list_iterations(lasif_root):
    """
    List iterations in project
    :param lasif_root: path to lasif root directory
    """

    comm = find_project_comm(lasif_root)

    iterations = comm.iterations.list()
    if len(iterations) == 0:
        print("There are no iterations in this project")
    else:
        if len(iterations) == 1:
            print(f"There is {len(iterations)} iteration in this project")
            print("Iteration known to LASIF: \n")
        else:
            print(f"There are {len(iterations)} iterations in this project")
            print("Iterations known to LASIF: \n")
    for iteration in iterations:
        print(comm.iterations.get_long_iteration_name(iteration), "\n")


def compare_misfits(lasif_root, from_it, to_it, events=[], weight_set=None,
                    print_events=False):
    """
    Compares the total misfit between two iterations.

    Total misfit is used regardless of the similarity of the picked windows
    from each iteration. This might skew the results but should
    give a good idea unless the windows change excessively between
    iterations.

    If windows are weighted in the calculation of the adjoint
    sources. That should translate into the calculated misfit
    value.
    :param lasif_root: path to lasif root directory
    :param from_it: evaluate misfit from this iteration
    :param to_it: to this iteration
    :param events: Events to use to compare
    :param weight_set: Set of station and event weights
    :param print_events: compare misfits for each event
    """
    comm = find_project_comm(lasif_root)

    if len(events) == 0:
        events = comm.events.list()

    if weight_set:
        if not comm.weights.has_weight_set(weight_set):
            raise LASIFNotFoundError(f"Weights {weight_set} not known"
                                     f"to LASIF")
    # Check if iterations exist
    if not comm.iterations.has_iteration(from_it):
            raise LASIFNotFoundError(f"Iteration {from_it} not known to LASIF")
    if not comm.iterations.has_iteration(to_it):
            raise LASIFNotFoundError(f"Iteration {to_it} not known to LASIF")

    from_it_misfit = 0.0
    to_it_misfit = 0.0
    for event in events:
        from_it_misfit += \
            comm.adj_sources.get_misfit_for_event(event,
                                                  from_it,
                                                  weight_set)
        to_it_misfit += \
            comm.adj_sources.get_misfit_for_event(event,
                                                  to_it,
                                                  weight_set)
        if print_events:
            # Print information about every event.
            from_it_misfit_event = \
                comm.adj_sources.get_misfit_for_event(event,
                                                      from_it,
                                                      weight_set)
            to_it_misfit_event = \
                comm.adj_sources.get_misfit_for_event(event,
                                                      to_it,
                                                      weight_set)
            print(f"{event}: \n"
                  f"\t iteration {from_it} has misfit: "
                  f"{from_it_misfit_event} \n"
                  f"\t iteration {to_it} has misfit: {to_it_misfit_event}.")
            rel_change = ((to_it_misfit_event - from_it_misfit_event) /
                          from_it_misfit_event * 100.0)
            print(f"Relative change: {rel_change:.2f}%")

    print(f"Total misfit for iteration {from_it}: {from_it_misfit}")
    print(f"Total misfit for iteration {to_it}: {to_it_misfit}")
    rel_change = (to_it_misfit - from_it_misfit) / from_it_misfit * 100.0
    print(f"Relative change in total misfit from iteration {from_it} to "
          f"{to_it} is: {rel_change:.2f}")
    n_events = len(comm.events.list())
    print(f"Misfit per event for iteration {from_it}: "
          f"{from_it_misfit/n_events}")
    print(f"Misfit per event for iteration {to_it}: "
          f"{to_it_misfit/n_events}")


def list_weight_sets(lasif_root):
    """
    Print a list of all weight sets in the project.
    :param lasif_root: path to lasif root directory
    """

    comm = find_project_comm(lasif_root)

    it_len = comm.weights.count()

    print("%i weight set(s)%s in project:" % (it_len,
                                              "s" if it_len != 1 else ""))
    for weights in comm.weights.list():
        print("\t%s" % weights)


def process_data(lasif_root, events=[], iteration=None):
    """
    Process recorded data
    :param lasif_root: path to lasif root directory
    :param events: Process data from specific events [optional]
    :param iteration: Process data from events used in an iteration [optional]
    """
    comm = find_project_comm(lasif_root)

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

    exceptions = []
    if MPI.COMM_WORLD.rank == 0:
        # Check if the event ids are valid.
        if not exceptions and events:
            for event_name in events:
                if not comm.events.has_event(event_name):
                    msg = "Event '%s' not found." % event_name
                    exceptions.append(msg)
                    break

    exceptions = MPI.COMM_WORLD.bcast(exceptions, root=0)
    if exceptions:
        raise Exception(exceptions[0])

    # Make sure all the ranks enter the processing at the same time.
    MPI.COMM_WORLD.barrier()
    comm.actions.process_data(events)


def plot_window_statistics(lasif_root, window_set, save=False, events=[],
                           iteration=None):
    """
    Plot some statistics related to windows in a specific set.
    :param lasif_root: path to lasif root directory
    :param window_set: name of window set
    :param save: Saves the plot in a file
    :param events: Plot statistics related to specific events [optional]
    :param iteration: Plot statistics related to events in a specific iteration
    [optional]
    """
    import matplotlib.pyplot as plt
    comm = find_project_comm(lasif_root)

    if len(events) == 0:
        events = comm.events.list(iteration=iteration)

    if save:
        plt.switch_backend('agg')

    if not comm.windows.has_window_set(window_set):
        raise LASIFNotFoundError("Could not find the specified window set")

    comm.visualizations.plot_window_statistics(
        window_set, events, ax=None, show=False)

    if save:
        outfile = os.path.join(
            comm.project.get_output_folder(
                type="window_statistics_plots", tag="windows",
                timestamp=False), f"{window_set}.png", )
        plt.savefig(outfile, dpi=200, transparent=True)
        print("Saved picture at %s" % outfile)
    else:
        plt.show()


def plot_windows(lasif_root, event_name, window_set, distance_bins=500):
    """
    Plot the selected windows for a specific event
    :param lasif_root: path to lasif root directory
    :param event_name: name of event
    :param window_set: name of window set
    :param distance_bins: number of bins on distance axis for combined plot
    [optional]
    """

    comm = find_project_comm(lasif_root)

    comm.visualizations.plot_windows(event=event_name,
                                     window_set_name=window_set,
                                     ax=None,
                                     distance_bins=500,
                                     show=True)


def write_stations_to_file(lasif_root):
    """
    This function writes a csv file with station name and lat,lon locations
    :param lasif_root: path to lasif root directory or communicator
    """
    import pandas as pd
    if isinstance(lasif_root, Communicator):
        comm = lasif_root
    else:
        comm = find_project_comm(lasif_root)

    events = comm.events.list()

    station_list = {"station": [], "latitude": [], "longitude": []}

    for event in events:
        try:
            data = comm.query.get_all_stations_for_event(event,
                                                         list_only=False)
        except LASIFNotFoundError:
            continue
        for key in data:
            if key not in station_list["station"]:
                station_list["station"].append(key)
                station_list["latitude"].append(data[key]['latitude'])
                station_list["longitude"].append(data[key]['latitude'])

    stations = pd.DataFrame(data=station_list)
    output_path = os.path.join(
        comm.project.paths["output"], "station_list.csv")
    stations.to_csv(path_or_buf=output_path, index=False)
    print(f"Wrote a list of stations to file: {output_path}")


def get_subset(lasif_root, events, count, existing_events=None):
    """
    This function gets an optimally distributed set of events
    :param comm: LASIF communicator
    :param count: number of events to choose.
    :param events: list of event_names, from which to choose from. These
    events must be known to LASIF
    :return:
    """
    from lasif.tools.query_gcmt_catalog import get_subset_of_events
    if isinstance(lasif_root, Communicator):
        comm = lasif_root
    else:
        comm = find_project_comm(lasif_root)
    return get_subset_of_events(comm, count, events, existing_events)


def validate_data(lasif_root, data_station_file_availability=False,
                  raypaths=False, full=False):
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

    :param lasif_root: path to lasif root directory
    :param data_station_file_availability: asserts that all stations have
    corresponding station files and all stations have waveforms, very slow.
    :param raypaths: Assert that all raypaths are within the set boundaries.
    Very slow.
    :param full: Run all validations
    """

    comm = find_project_comm(lasif_root)

    # If full check, check everything.
    if full:
        data_station_file_availability = True
        raypaths = True

    comm.validator.validate_data(
        data_and_station_file_availability=data_station_file_availability,
        raypaths=raypaths)


def clean_up(lasif_root, clean_up_file):
    """
    Clean up the lasif project. The required file can be created with
    the validate_data command.

    :param lasif_root: path to lasif root directory
    :param clean_up_file: path to clean-up file
    """

    comm = find_project_comm(lasif_root)
    if not os.path.exists(clean_up_file):
        raise LASIFNotFoundError(f"Could not find {clean_up_file}\n"
                                 f"Please check that the specified file path "
                                 f"is correct.")

    comm.validator.clean_up_project(clean_up_file)


def update_catalog():
    """
    Update GCMT catalog
    """
    from lasif.tools.query_gcmt_catalog import update_GCMT_catalog
    update_GCMT_catalog()


def tutorial():
    """
    Open the lasif tutorial in a web-browser.
    """

    import webbrowser
    webbrowser.open("http://dirkphilip.github.io/LASIF_2.0/")
