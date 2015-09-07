#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import itertools
import numpy as np
import os
import warnings

from lasif import LASIFError, LASIFWarning, LASIFNotFoundError
from lasif import rotations
from .component import Component


class ActionsComponent(Component):
    """
    Component implementing actions on the data. Requires most other
    components to be available.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def preprocess_data(self, iteration_name, event_names=None):
        """
        Preprocesses all data for a given iteration.

        This function works with and without MPI.

        :param event_names: event_ids is a list of events to process in this
            run. It will process all events if not given.
        """
        from mpi4py import MPI
        from lasif.tools.parallel_helpers import distribute_across_ranks

        iteration = self.comm.iterations.get(iteration_name)

        process_params = iteration.get_process_params()
        processing_tag = iteration.processing_tag

        def processing_data_generator():
            """
            Generate a dictionary with information for processing for each
            waveform.
            """
            # Loop over the chosen events.
            for event_name, event in iteration.events.iteritems():
                # None means to process all events, otherwise it will be a list
                # of events.
                if not ((event_names is None) or (event_name in event_names)):
                    continue

                output_folder = self.comm.waveforms.get_waveform_folder(
                    event_name=event_name, data_type="processed",
                    tag_or_iteration=processing_tag)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Get the event.
                event = self.comm.events.get(event_name)

                try:
                    # Get the stations.
                    stations = self.comm.query\
                        .get_all_stations_for_event(event_name)
                    # Get the raw waveform data.
                    waveforms = \
                        self.comm.waveforms.get_metadata_raw(event_name)
                except LASIFNotFoundError:
                    warnings.warn(
                        "No data found for event '%s'. Did you delete data "
                        "after the iteration has been created?" % event_name)
                    continue

                # Group by station name.
                def func(x):
                    return ".".join(x["channel_id"].split(".")[:2])

                waveforms.sort(key=func)
                for station_name, channels in  \
                        itertools.groupby(waveforms, func):
                    channels = list(channels)
                    # Filter waveforms with no available station files
                    # or coordinates.
                    if station_name not in stations:
                        continue

                    # Group by location.
                    def get_loc_id(x):
                        return x["channel_id"].split(".")[2]

                    channels.sort(key=get_loc_id)
                    locations = []
                    for loc_id, chans in itertools.groupby(channels,
                                                           get_loc_id):
                        locations.append((loc_id, list(chans)))
                    locations.sort(key=lambda x: x[0])

                    if len(locations) > 1:
                        msg = ("More than one location found for event "
                               "'%s' at station '%s'. The alphabetically "
                               "first one will be chosen." %
                               (event_name, station_name))
                        warnings.warn(msg, LASIFWarning)
                    location = locations[0][1]

                    # Loop over each found channel.
                    for channel in location:
                        channel.update(stations[station_name])
                        input_filename = channel["filename"]
                        output_filename = os.path.join(
                            output_folder,
                            os.path.basename(input_filename))
                        # Skip already processed files.
                        if os.path.exists(output_filename):
                            continue

                        ret_dict = {
                            "process_params": process_params,
                            "input_filename": input_filename,
                            "output_filename": output_filename,
                            "station_coordinates": {
                                "latitude": channel["latitude"],
                                "longitude": channel["longitude"],
                                "elevation_in_m": channel["elevation_in_m"],
                                "local_depth_in_m": channel[
                                    "local_depth_in_m"],
                            },
                            "station_filename": self.comm.stations.
                            get_channel_filename(channel["channel_id"],
                                                 channel["starttime"]),
                            "event_information": event,
                        }
                        yield ret_dict

        # Only rank 0 needs to know what has to be processsed.
        if MPI.COMM_WORLD.rank == 0:
            to_be_processed = [{"processing_info": _i, "iteration": iteration}
                               for _i in processing_data_generator()]
        else:
            to_be_processed = None

        # Load project specific window selection function.
        preprocessing_function = self.comm.project.get_project_function(
            "preprocessing_function")

        logfile = self.comm.project.get_log_file(
            "DATA_PREPROCESSING", "processing_iteration_%s" % (str(
                iteration.name)))

        distribute_across_ranks(
            function=preprocessing_function, items=to_be_processed,
            get_name=lambda x: x["processing_info"]["input_filename"],
            logfile=logfile)

    def select_windows(self, event, iteration):
        """
        Automatically select the windows for the given event and iteration.

        Will only attempt to select windows for stations that have no
        windows. Each station that has a window is assumed to have already
        been picked in some fashion.

        Function can be called with and without MPI.

        :param event: The event.
        :param iteration: The iteration.
        """
        from lasif.utils import channel2station
        from mpi4py import MPI

        event = self.comm.events.get(event)
        iteration = self.comm.iterations.get(iteration)

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

        # Only rank 0 needs to know what has to be processsed.
        if MPI.COMM_WORLD.rank == 0:
            # All stations for the given iteration and event.
            stations = \
                set(iteration.events[event["event_name"]]["stations"].keys())

            # Get all stations that currently do not have windows.
            windows = self.comm.windows.get(event, iteration).list()
            stations_without_windows = \
                stations - set(map(channel2station, windows))
            total_size = len(stations_without_windows)
            stations_without_windows = split(list(stations_without_windows),
                                             MPI.COMM_WORLD.size)

            # Initialize station cache on rank 0.
            self.comm.stations.file_count
            # Also initialize the processed and synthetic data caches. They
            # have to exist before the other ranks can access them.
            try:
                self.comm.waveforms.get_waveform_cache(
                    event["event_name"], "processed", iteration.processing_tag)
            except LASIFNotFoundError:
                pass
            try:
                self.comm.waveforms.get_waveform_cache(
                    event["event_name"], "synthetic", iteration)
            except LASIFNotFoundError:
                pass
        else:
            stations_without_windows = None

        # Distribute on a per-station basis.
        stations_without_windows = MPI.COMM_WORLD.scatter(
            stations_without_windows, root=0)

        for _i, station in enumerate(stations_without_windows):
            try:
                self.select_windows_for_station(event, iteration, station)
            except LASIFNotFoundError as e:
                warnings.warn(str(e), LASIFWarning)
            except Exception as e:
                warnings.warn(
                    "Exception occured for iteration %s, event %s, and "
                    "station %s: %s" % (iteration.name, event["event_name"],
                                        station, str(e)), LASIFWarning)
            if MPI.COMM_WORLD.rank == 0:
                print("Window picking process: Picked windows for approx. %i "
                      "of %i stations." % (
                          min(_i * MPI.COMM_WORLD.size, total_size),
                          total_size))

        # Barrier at the end useful for running this in a loop.
        MPI.COMM_WORLD.barrier()

    def select_windows_for_station(self, event, iteration, station, **kwargs):
        """
        Selects windows for the given event, iteration, and station. Will
        delete any previously existing windows for that station if any.

        :param event: The event.
        :param iteration: The iteration.
        :param station: The station id in the form NET.STA.
        """
        from lasif.utils import select_component_from_stream

        # Load project specific window selection function.
        select_windows = self.comm.project.get_project_function(
            "window_picking_function")

        event = self.comm.events.get(event)
        iteration = self.comm.iterations.get(iteration)
        data = self.comm.query.get_matching_waveforms(event, iteration,
                                                      station)

        process_params = iteration.get_process_params()
        minimum_period = 1.0 / process_params["lowpass"]
        maximum_period = 1.0 / process_params["highpass"]

        window_group_manager = self.comm.windows.get(event, iteration)
        # Delete the windows for this stations.
        window_group_manager.delete_windows_for_station(station)

        found_something = False
        for component in ["E", "N", "Z"]:
            try:
                data_tr = select_component_from_stream(data.data, component)
                synth_tr = select_component_from_stream(data.synthetics,
                                                        component)
            except LASIFNotFoundError:
                continue
            found_something = True

            windows = select_windows(data_tr, synth_tr, event["latitude"],
                                     event["longitude"], event["depth_in_km"],
                                     data.coordinates["latitude"],
                                     data.coordinates["longitude"],
                                     minimum_period=minimum_period,
                                     maximum_period=maximum_period,
                                     iteration=iteration, **kwargs)
            if not windows:
                continue

            window_group = window_group_manager.get(data_tr.id)
            for starttime, endtime in windows:
                window_group.add_window(starttime=starttime, endtime=endtime)
            window_group.write()

        if found_something is False:
            raise LASIFNotFoundError(
                "No matching data found for event '%s', iteration '%s', and "
                "station '%s'." % (event["event_name"], iteration.name,
                                   station))

    def generate_input_files(self, iteration_name, event_name,
                             simulation_type):
        """
        Generate the input files for one event.

        :param iteration_name: The name of the iteration.
        :param event_name: The name of the event for which to generate the
            input files.
        :param simulate_type: The type of simulation to perform. Possible
            values are: 'normal simulate', 'adjoint forward', 'adjoint
            reverse'
        """
        from wfs_input_generator import InputFileGenerator

        # =====================================================================
        # read iteration xml file, get event and list of stations
        # =====================================================================

        iteration = self.comm.iterations.get(iteration_name)

        # Check that the event is part of the iterations.
        if event_name not in iteration.events:
            msg = ("Event '%s' not part of iteration '%s'.\nEvents available "
                   "in iteration:\n\t%s" %
                   (event_name, iteration_name, "\n\t".join(
                       sorted(iteration.events.keys()))))
            raise ValueError(msg)

        event = self.comm.events.get(event_name)
        stations_for_event = iteration.events[event_name]["stations"].keys()

        # Get all stations and create a dictionary for the input file
        # generator.
        stations = self.comm.query.get_all_stations_for_event(event_name)
        stations = [{"id": key, "latitude": value["latitude"],
                     "longitude": value["longitude"],
                     "elevation_in_m": value["elevation_in_m"],
                     "local_depth_in_m": value["local_depth_in_m"]}
                    for key, value in stations.iteritems()
                    if key in stations_for_event]

        # =====================================================================
        # set solver options
        # =====================================================================

        solver = iteration.solver_settings

        # Currently only SES3D 4.1 is supported
        solver_format = solver["solver"].lower()
        if solver_format not in ["ses3d 4.1", "ses3d 2.0",
                                 "specfem3d cartesian", "specfem3d globe cem"]:
            msg = ("Currently only SES3D 4.1, SES3D 2.0, SPECFEM3D "
                   "CARTESIAN, and SPECFEM3D GLOBE CEM are supported.")
            raise ValueError(msg)
        solver_format = solver_format.replace(' ', '_')
        solver_format = solver_format.replace('.', '_')

        solver = solver["solver_settings"]

        # =====================================================================
        # create the input file generator, add event and stations,
        # populate the configuration items
        # =====================================================================

        # Add the event and the stations to the input file generator.
        gen = InputFileGenerator()
        gen.add_events(event["filename"])
        gen.add_stations(stations)

        if solver_format in ["ses3d_4_1", "ses3d_2_0"]:
            # event tag
            gen.config.event_tag = event_name

            # Time configuration.
            npts = solver["simulation_parameters"]["number_of_time_steps"]
            delta = solver["simulation_parameters"]["time_increment"]
            gen.config.number_of_time_steps = npts
            gen.config.time_increment_in_s = delta

            # SES3D specific configuration
            gen.config.output_folder = solver["output_directory"].replace(
                "{{EVENT_NAME}}", event_name.replace(" ", "_"))
            gen.config.simulation_type = simulation_type

            gen.config.adjoint_forward_wavefield_output_folder = \
                solver["adjoint_output_parameters"][
                    "forward_field_output_directory"].replace(
                    "{{EVENT_NAME}}", event_name.replace(" ", "_"))
            gen.config.adjoint_forward_sampling_rate = \
                solver["adjoint_output_parameters"][
                    "sampling_rate_of_forward_field"]

            # Visco-elastic dissipation
            diss = solver["simulation_parameters"]["is_dissipative"]
            gen.config.is_dissipative = diss

            # Only SES3D 4.1 has the relaxation parameters.
            if solver_format == "ses3d_4_1":
                gen.config.Q_model_relaxation_times = \
                    solver["relaxation_parameter_list"]["tau"]
                gen.config.Q_model_weights_of_relaxation_mechanisms = \
                    solver["relaxation_parameter_list"]["w"]

            # Discretization
            disc = solver["computational_setup"]
            gen.config.nx_global = disc["nx_global"]
            gen.config.ny_global = disc["ny_global"]
            gen.config.nz_global = disc["nz_global"]
            gen.config.px = disc["px_processors_in_theta_direction"]
            gen.config.py = disc["py_processors_in_phi_direction"]
            gen.config.pz = disc["pz_processors_in_r_direction"]
            gen.config.lagrange_polynomial_degree = \
                disc["lagrange_polynomial_degree"]

            # Configure the mesh.
            domain = self.comm.project.domain
            gen.config.mesh_min_latitude = domain.min_latitude
            gen.config.mesh_max_latitude = domain.max_latitude
            gen.config.mesh_min_longitude = domain.min_longitude
            gen.config.mesh_max_longitude = domain.max_longitude
            gen.config.mesh_min_depth_in_km = domain.min_depth_in_km
            gen.config.mesh_max_depth_in_km = domain.max_depth_in_km

            # Set the rotation parameters.
            gen.config.rotation_angle_in_degree = \
                domain.rotation_angle_in_degree
            gen.config.rotation_axis = domain.rotation_axis

            # Make source time function
            gen.config.source_time_function = \
                iteration.get_source_time_function()["data"]
        elif solver_format == "specfem3d_cartesian":
            gen.config.NSTEP = \
                solver["simulation_parameters"]["number_of_time_steps"]
            gen.config.DT = \
                solver["simulation_parameters"]["time_increment"]
            gen.config.NPROC = \
                solver["computational_setup"]["number_of_processors"]
            if simulation_type == "normal simulation":
                msg = ("'normal_simulate' not supported for SPECFEM3D "
                       "Cartesian. Please choose either 'adjoint_forward' or "
                       "'adjoint_reverse'.")
                raise NotImplementedError(msg)
            elif simulation_type == "adjoint forward":
                gen.config.SIMULATION_TYPE = 1
            elif simulation_type == "adjoint reverse":
                gen.config.SIMULATION_TYPE = 2
            else:
                raise NotImplementedError
            solver_format = solver_format.upper()

        elif solver_format == "specfem3d_globe_cem":
            cs = solver["computational_setup"]
            gen.config.NPROC_XI = cs["number_of_processors_xi"]
            gen.config.NPROC_ETA = cs["number_of_processors_eta"]
            gen.config.NCHUNKS = cs["number_of_chunks"]
            gen.config.NEX_XI = cs["elements_per_chunk_xi"]
            gen.config.NEX_ETA = cs["elements_per_chunk_eta"]
            gen.config.OCEANS = cs["simulate_oceans"]
            gen.config.ELLIPTICITY = cs["simulate_ellipticity"]
            gen.config.TOPOGRAPHY = cs["simulate_topography"]
            gen.config.GRAVITY = cs["simulate_gravity"]
            gen.config.ROTATION = cs["simulate_rotation"]
            gen.config.ATTENUATION = cs["simulate_attenuation"]
            gen.config.ABSORBING_CONDITIONS = True
            if cs["fast_undo_attenuation"]:
                gen.config.PARTIAL_PHYS_DISPERSION_ONLY = True
                gen.config.UNDO_ATTENUATION = False
            else:
                gen.config.PARTIAL_PHYS_DISPERSION_ONLY = False
                gen.config.UNDO_ATTENUATION = True
            gen.config.GPU_MODE = cs["use_gpu"]
            gen.config.SOURCE_TIME_FUNCTION = \
                iteration.get_source_time_function()["data"]

            if simulation_type == "normal simulation":
                gen.config.SIMULATION_TYPE = 1
                gen.config.SAVE_FORWARD = False
            elif simulation_type == "adjoint forward":
                gen.config.SIMULATION_TYPE = 1
                gen.config.SAVE_FORWARD = True
            elif simulation_type == "adjoint reverse":
                gen.config.SIMULATION_TYPE = 2
                gen.config.SAVE_FORWARD = True
            else:
                raise NotImplementedError

            # Use the current domain setting to derive the bounds in the way
            # SPECFEM specifies them.
            domain = self.comm.project.domain

            lat_range = domain.max_latitude - \
                domain.min_latitude
            lng_range = domain.max_longitude - \
                domain.min_longitude

            c_lat = \
                domain.min_latitude + lat_range / 2.0
            c_lng = \
                domain.min_longitude + lng_range / 2.0

            # Rotate the point.
            c_lat_1, c_lng_1 = rotations.rotate_lat_lon(
                c_lat, c_lng, domain.rotation_axis,
                domain.rotation_angle_in_degree)

            # SES3D rotation.
            A = rotations._get_rotation_matrix(
                domain.rotation_axis, domain.rotation_angle_in_degree)

            latitude_rotation = -(c_lat_1 - c_lat)
            longitude_rotation = c_lng_1 - c_lng

            # Rotate the latitude. The rotation axis is latitude 0 and
            # the center longitude + 90 degree
            B = rotations._get_rotation_matrix(
                rotations.lat_lon_radius_to_xyz(0.0, c_lng + 90, 1.0),
                latitude_rotation)
            # Rotate around the North pole.
            C = rotations._get_rotation_matrix(
                [0.0, 0.0, 1.0], longitude_rotation)

            D = A * np.linalg.inv(C * B)

            axis, angle = rotations._get_axis_and_angle_from_rotation_matrix(D)
            rotated_axis = rotations.xyz_to_lat_lon_radius(*axis)

            # Consistency check
            if abs(rotated_axis[0] - c_lat_1) >= 0.01 or \
                    abs(rotated_axis[1] - c_lng_1) >= 0.01:
                axis *= -1.0
                angle *= -1.0
                rotated_axis = rotations.xyz_to_lat_lon_radius(*axis)

            if abs(rotated_axis[0] - c_lat_1) >= 0.01 or \
                    abs(rotated_axis[1] - c_lng_1) >= 0.01:
                msg = "Failed to describe the domain in terms that SPECFEM " \
                      "understands. The domain definition in the output " \
                      "files will NOT BE CORRECT!"
                warnings.warn(msg, LASIFWarning)

            gen.config.ANGULAR_WIDTH_XI_IN_DEGREES = lng_range
            gen.config.ANGULAR_WIDTH_ETA_IN_DEGREES = lat_range
            gen.config.CENTER_LATITUDE_IN_DEGREES = c_lat_1
            gen.config.CENTER_LONGITUDE_IN_DEGREES = c_lng_1
            gen.config.GAMMA_ROTATION_AZIMUTH = angle

            gen.config.MODEL = cs["model"]

            pp = iteration.get_process_params()
            gen.config.RECORD_LENGTH_IN_MINUTES = \
                (pp["npts"] * pp["dt"]) / 60.0
            solver_format = solver_format.upper()

        else:
            msg = "Unknown solver '%s'." % solver_format
            raise NotImplementedError(msg)

        # =================================================================
        # output
        # =================================================================
        output_dir = self.comm.project.get_output_folder(
            type="input_files",
            tag="ITERATION_%s__%s__EVENT_%s" % (
                iteration_name, simulation_type.replace(" ", "_"),
                event_name))

        gen.write(format=solver_format, output_dir=output_dir)
        print "Written files to '%s'." % output_dir

    def calculate_all_adjoint_sources(self, iteration_name, event_name):
        """
        Function to calculate all adjoint sources for a certain iteration
        and event.
        """
        window_manager = self.comm.windows.get(event_name, iteration_name)
        event = self.comm.events.get(event_name)
        iteration = self.comm.iterations.get(iteration_name)
        iteration_event_def = iteration.events[event["event_name"]]
        iteration_stations = iteration_event_def["stations"]

        l = sorted(window_manager.list())
        for station, windows in itertools.groupby(
                l, key=lambda x: ".".join(x.split(".")[:2])):
            if station not in iteration_stations:
                continue
            try:
                for w in windows:
                    w = window_manager.get(w)
                    for window in w:
                        # Access the property will trigger an adjoint source
                        # calculation.
                        window.adjoint_source
            except LASIFError as e:
                print("Could not calculate adjoint source for iteration %s "
                      "and station %s. Repick windows? Reason: %s" % (
                          iteration.name, station, str(e)))

    def finalize_adjoint_sources(self, iteration_name, event_name):
        """
        Finalizes the adjoint sources.
        """
        from itertools import izip
        import numpy as np
        from lasif import rotations

        window_manager = self.comm.windows.get(event_name, iteration_name)
        event = self.comm.events.get(event_name)
        iteration = self.comm.iterations.get(iteration_name)
        iteration_event_def = iteration.events[event["event_name"]]
        iteration_stations = iteration_event_def["stations"]

        # For now assume that the adjoint sources have the same
        # sampling rate as the synthetics which in LASIF's workflow
        # actually has to be true.
        dt = iteration.get_process_params()["dt"]

        # Current domain and solver.
        domain = self.comm.project.domain
        solver = iteration.solver_settings["solver"].lower()

        adjoint_source_stations = set()

        if "ses3d" in solver:
            ses3d_all_coordinates = []

        event_weight = iteration_event_def["event_weight"]

        output_folder = self.comm.project.get_output_folder(
            type="adjoint_sources",
            tag="ITERATION_%s__%s" % (iteration_name, event_name))

        l = sorted(window_manager.list())
        for station, windows in itertools.groupby(
                l, key=lambda x: ".".join(x.split(".")[:2])):
            if station not in iteration_stations:
                continue
            print ".",
            station_weight = iteration_stations[station]["station_weight"]
            channels = {}
            try:
                for w in windows:
                    w = window_manager.get(w)
                    channel_weight = 0
                    srcs = []
                    for window in w:
                        ad_src = window.adjoint_source
                        if not ad_src["adjoint_source"].ptp():
                            continue
                        srcs.append(ad_src["adjoint_source"] * window.weight)
                        channel_weight += window.weight
                    if not srcs:
                        continue
                    # Final adjoint source for that channel and apply all
                    # weights.
                    adjoint_source = np.sum(srcs, axis=0) / channel_weight * \
                        event_weight * station_weight
                    channels[w.channel_id[-1]] = adjoint_source
            except LASIFError as e:
                print("Could not calculate adjoint source for iteration %s "
                      "and station %s. Repick windows? Reason: %s" % (
                          iteration.name, station, str(e)))
                continue
            if not channels:
                continue
            # Now all adjoint sources of a window should have the same length.
            length = set(len(v) for v in channels.values())
            assert len(length) == 1
            length = length.pop()
            # All missing channels will be replaced with a zero array.
            for c in ["Z", "N", "E"]:
                if c in channels:
                    continue
                channels[c] = np.zeros(length)

            # Get the station coordinates
            coords = self.comm.query.get_coordinates_for_station(event_name,
                                                                 station)

            # Rotate. if needed
            rec_lat = coords["latitude"]
            rec_lng = coords["longitude"]

            # The adjoint sources depend on the solver.
            if "ses3d" in solver:
                # Rotate if needed.
                if domain.rotation_angle_in_degree:
                    # Rotate the adjoint source location.
                    r_rec_lat, r_rec_lng = rotations.rotate_lat_lon(
                        rec_lat, rec_lng, domain.rotation_axis,
                        -domain.rotation_angle_in_degree)
                    # Rotate the adjoint sources.
                    channels["N"], channels["E"], channels["Z"] = \
                        rotations.rotate_data(
                            channels["N"], channels["E"],
                            channels["Z"], rec_lat, rec_lng,
                            domain.rotation_axis,
                            -domain.rotation_angle_in_degree)
                else:
                    r_rec_lat = rec_lat
                    r_rec_lng = rec_lng
                r_rec_depth = 0.0
                r_rec_colat = rotations.lat2colat(r_rec_lat)

                # Now once again map from ZNE to the XYZ of SES3D.
                CHANNEL_MAPPING = {"X": "N", "Y": "E", "Z": "Z"}
                adjoint_source_stations.add(station)
                adjoint_src_filename = os.path.join(
                    output_folder, "ad_src_%i" % len(adjoint_source_stations))
                ses3d_all_coordinates.append(
                    (r_rec_colat, r_rec_lng, r_rec_depth))

                # Actually write the adjoint source file in SES3D specific
                # format.
                with open(adjoint_src_filename, "wt") as open_file:
                    open_file.write("-- adjoint source ------------------\n")
                    open_file.write(
                        "-- source coordinates (colat,lon,depth)\n")
                    open_file.write("%f %f %f\n" % (r_rec_colat, r_rec_lng,
                                                    r_rec_depth))
                    open_file.write("-- source time function (x, y, z) --\n")
                    # Revert the X component as it has to point south in SES3D.
                    for x, y, z in izip(-1.0 * channels[CHANNEL_MAPPING["X"]],
                                        channels[CHANNEL_MAPPING["Y"]],
                                        channels[CHANNEL_MAPPING["Z"]]):
                        open_file.write("%e %e %e\n" % (x, y, z))
                    open_file.write("\n")
            elif "specfem" in solver:
                s_set = iteration.solver_settings["solver_settings"]
                if "adjoint_source_time_shift" not in s_set:
                    warnings.warn("No <adjoint_source_time_shift> tag in the "
                                  "iteration XML file. No time shift for the "
                                  "adjoint sources will be applied.",
                                  LASIFWarning)
                    src_time_shift = 0
                else:
                    src_time_shift = float(s_set["adjoint_source_time_shift"])
                adjoint_source_stations.add(station)
                # Write all components. The adjoint sources right now are
                # not time shifted.
                for component in ["Z", "N", "E"]:
                    # XXX: M band code could be different.
                    adjoint_src_filename = os.path.join(
                        output_folder, "%s.MX%s.adj" % (station, component))
                    adj_src = channels[component]
                    l = len(adj_src)
                    to_write = np.empty((l, 2))
                    to_write[:, 0] = \
                        np.linspace(0, (l - 1) * dt, l) + src_time_shift

                    # SPECFEM expects non-time reversed adjoint sources and
                    # the sign is different for some reason.
                    to_write[:, 1] = -1.0 * adj_src[::-1]

                    np.savetxt(adjoint_src_filename, to_write)
            else:
                raise NotImplementedError(
                    "Adjoint source writing for solver '%s' not yet "
                    "implemented." % iteration.solver_settings["solver"])

        if not adjoint_source_stations:
            print("Could not create a single adjoint source.")
            return

        if "ses3d" in solver:
            with open(os.path.join(output_folder, "ad_srcfile"), "wt") as fh:
                fh.write("%i\n" % len(adjoint_source_stations))
                for line in ses3d_all_coordinates:
                    fh.write("%.6f %.6f %.6f\n" % (line[0], line[1], line[2]))
                fh.write("\n")
        elif "specfem" in solver:
            adjoint_source_stations = sorted(list(adjoint_source_stations))
            with open(os.path.join(output_folder, "STATIONS_ADJOINT"),
                      "wt") as fh:
                for station in adjoint_source_stations:
                    coords = self.comm.query.get_coordinates_for_station(
                        event_name, station)
                    fh.write("{sta} {net} {lat} {lng} {ele} {dep}\n".format(
                        sta=station.split(".")[1],
                        net=station.split(".")[0],
                        lat=coords["latitude"],
                        lng=coords["longitude"],
                        ele=coords["elevation_in_m"],
                        dep=coords["local_depth_in_m"]))

        print "Wrote adjoint sources for %i station(s) to %s." % (
            len(adjoint_source_stations), os.path.relpath(output_folder))
