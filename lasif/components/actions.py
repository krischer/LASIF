#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import itertools
import numpy as np
import os
import warnings

from lasif import LASIFError, LASIFWarning
from lasif import rotations
from .component import Component


class ActionsComponent(Component):
    """
    Component implementing actions on the data. Requires most other
    components to be available.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def preprocess_data(self, iteration_name, event_names=None,
                        waiting_time=4.0):
        """
        Preprocesses all data for a given iteration.

        :param waiting_time: The time spent sleeping after the initial message
            has been printed. Useful if the user should be given the chance to
            cancel the processing.
        :param event_names: event_ids is a list of events to process in this
            run. It will process all events if not given.
        """
        import colorama
        from lasif import preprocessing

        iteration = self.comm.iterations.get(iteration_name)

        process_params = iteration.get_process_params()
        processing_tag = iteration.processing_tag

        logfile = os.path.join(
            self.comm.project.get_output_folder("data_preprocessing"),
            "log.txt")

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
                # Get the stations.
                stations = self.comm.query\
                    .get_all_stations_for_event(event_name)
                # Get the raw waveform data.
                waveforms = \
                    self.comm.waveforms.get_metadata_raw(event_name)

                # Group by station name.
                func = lambda x: ".".join(x["channel_id"].split(".")[:2])
                for station_name, channels in  \
                        itertools.groupby(waveforms, func):
                    # Filter waveforms with no available station files
                    # or coordinates.
                    if station_name not in stations:
                        continue
                    # Group by location.
                    locations = list(itertools.groupby(
                        channels, lambda x: x["channel_id"].split(".")[2]))
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

        file_count = preprocessing.launch_processing(
            processing_data_generator(), log_filename=logfile,
            waiting_time=waiting_time, process_params=process_params)

        print("\nFinished processing %i files." %
              file_count["total_file_count"])
        if file_count["failed_file_count"]:
            print("\t%s%i files failed being processed.%s" %
                  (colorama.Fore.RED, file_count["failed_file_count"],
                   colorama.Fore.RESET))
        if file_count["warning_file_count"]:
            print("\t%s%i files raised warnings while being processed.%s" %
                  (colorama.Fore.YELLOW, file_count["warning_file_count"],
                   colorama.Fore.RESET))
        print("\t%s%i files have been processed without errors or warnings%s" %
              (colorama.Fore.GREEN, file_count["successful_file_count"],
               colorama.Fore.RESET))

        print("Logfile written to '%s'." % os.path.relpath(logfile))

    def select_windows(self, event, iteration):
        """
        Automatically select the windows for the given event and iteration.

        :param event: The event.
        :param iteration: The iteration.
        """
        event = self.comm.events.get(event)
        iteration = self.comm.iterations.get(iteration)


        from IPython.core.debugger import Tracer; Tracer(colors="Linux")()



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
            npts = solver["simulate_parameters"]["number_of_time_steps"]
            delta = solver["simulate_parameters"]["time_increment"]
            gen.config.number_of_time_steps = npts
            gen.config.time_increment_in_s = delta

            # SES3D specific configuration
            gen.config.output_folder = solver["output_directory"].replace(
                "{{EVENT_NAME}}", event_name.replace(" ", "_"))
            gen.config.simulate_type = simulation_type

            gen.config.adjoint_forward_wavefield_output_folder = \
                solver["adjoint_output_parameters"][
                    "forward_field_output_directory"].replace(
                    "{{EVENT_NAME}}", event_name.replace(" ", "_"))
            gen.config.adjoint_forward_sampling_rate = \
                solver["adjoint_output_parameters"][
                    "sampling_rate_of_forward_field"]

            # Visco-elastic dissipation
            diss = solver["simulate_parameters"]["is_dissipative"]
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
            gen.config.mesh_min_latitude = \
                domain["bounds"]["minimum_latitude"]
            gen.config.mesh_max_latitude = \
                domain["bounds"]["maximum_latitude"]
            gen.config.mesh_min_longitude = \
                domain["bounds"]["minimum_longitude"]
            gen.config.mesh_max_longitude = \
                domain["bounds"]["maximum_longitude"]
            gen.config.mesh_min_depth_in_km = \
                domain["bounds"]["minimum_depth_in_km"]
            gen.config.mesh_max_depth_in_km = \
                domain["bounds"]["maximum_depth_in_km"]

            # Set the rotation parameters.
            gen.config.rotation_angle_in_degree = domain["rotation_angle"]
            gen.config.rotation_axis = domain["rotation_axis"]

            # Make source time function
            gen.config.source_time_function = \
                iteration.get_source_time_function()["data"]
        elif solver_format == "specfem3d_cartesian":
            gen.config.NSTEP = \
                solver["simulate_parameters"]["number_of_time_steps"]
            gen.config.DT = \
                solver["simulate_parameters"]["time_increment"]
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

            lat_range = domain["bounds"]["maximum_latitude"] - \
                domain["bounds"]["minimum_latitude"]
            lng_range = domain["bounds"]["maximum_longitude"] - \
                        domain["bounds"]["minimum_longitude"]

            c_lat = \
                domain["bounds"]["minimum_latitude"] + lat_range / 2.0
            c_lng = \
                domain["bounds"]["minimum_longitude"] + lng_range / 2.0

            # Rotate the point.
            c_lat_1, c_lng_1 = rotations.rotate_lat_lon(
                c_lat, c_lng, domain["rotation_axis"],
                domain["rotation_angle"])

            # SES3D rotation.
            A = rotations._get_rotation_matrix(
                domain["rotation_axis"], domain["rotation_angle"])

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
                      "understands"
                raise LASIFError(msg)

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
            "input_files___ITERATION_%s__%s__EVENT_%s" % (
                iteration_name, simulation_type.replace(" ", "_"),
                event_name))

        gen.write(format=solver_format, output_dir=output_dir)
        print "Written files to '%s'." % output_dir
