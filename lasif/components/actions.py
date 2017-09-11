#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import itertools
import numpy as np
import os
import warnings

from lasif import LASIFError, LASIFWarning, LASIFNotFoundError

from .component import Component


class ActionsComponent(Component):
    """
    Component implementing actions on the data. Requires most other
    components to be available.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def preprocess_data(self, weight_set_name, event_names=None):
        """
        Preprocesses all data for a given iteration.

        This function works with and without MPI.

        :param event_names: event_ids is a list of events to process in this
            run. It will process all events if not given.
        """
        from mpi4py import MPI
        weight_set = self.comm.weights.get(weight_set_name)
        process_params = self.comm.project.preprocessing_params
        solver_params = self.comm.project.solver

        simulation_params = solver_params["settings"]["simulation_parameters"]
        npts = simulation_params["number_of_time_steps"]
        sampling_rate = simulation_params["sampling_rate"]
        dt = simulation_params["time_increment"]
        salvus_start_time = simulation_params["start_time"]

        def processing_data_generator():
            """
            Generate a dictionary with information for processing for each
            waveform.
            """
            # Loop over the chosen events.
            for event_name, event in weight_set.events.items():
                # None means to process all events, otherwise it will be a list
                # of events.
                if not ((event_names is None) or (event_name in event_names)):
                    continue

                output_folder = os.path.join(self.comm.project.paths["preproc_eq_data"], event_name)

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Test if event is available.
                event = self.comm.events.get(event_name)

                asdf_file_name = self.comm.waveforms.get_asdf_filename(event_name, data_type="raw")
                lowpass_period = process_params["lowpass_period"]
                highpass_period = process_params["highpass_period"]
                preprocessing_tag = self.comm.waveforms.preprocessing_tag

                output_filename = os.path.join(output_folder, preprocessing_tag + ".h5")

                # remove asdf file if it already exists
                if MPI.COMM_WORLD.rank == 0:
                    if os.path.exists(output_filename):
                        os.remove(output_filename)

                ret_dict = {
                    "process_params": process_params,
                    "asdf_input_filename": asdf_file_name,
                    "asdf_output_filename": output_filename,
                    "preprocessing_tag": preprocessing_tag,
                    "dt": dt,
                    "npts": npts,
                    "sampling_rate": sampling_rate,
                    "salvus_start_time": salvus_start_time,
                    "lowpass_period": lowpass_period,
                    "highpass_period": highpass_period
                }
                yield ret_dict


        to_be_processed = [{"processing_info": _i}
                           for _i in processing_data_generator()]

        # Load project specific window selection function.
        preprocessing_function_asdf = self.comm.project.get_project_function(
            "preprocessing_function_asdf")

        MPI.COMM_WORLD.Barrier()
        for event in to_be_processed:
            preprocessing_function_asdf(event["processing_info"])
            MPI.COMM_WORLD.Barrier()



    def calculate_adjoint_sources(self, event, iteration, **kwargs):
        from lasif.utils import select_component_from_stream

        from mpi4py import MPI
        import pyasdf

        event = self.comm.events.get(event)
        iteration = self.comm.iterations.get(iteration)

        # Get the ASDF filenames.
        processed_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="processed",
            tag_or_iteration=iteration.processing_tag)
        synthetic_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="synthetic",
            tag_or_iteration=iteration.name)

        if not os.path.exists(processed_filename):
            msg = "File '%s' does not exists." % processed_filename
            raise LASIFNotFoundError(msg)

        if not os.path.exists(synthetic_filename):
            msg = "File '%s' does not exists." % synthetic_filename
            raise LASIFNotFoundError(msg)

        # Read all windows on rank 0 and broadcast.
        if MPI.COMM_WORLD.rank == 0:
            all_windows = self.comm.wins_and_adj_sources.read_all_windows(
                event=event, iteration=iteration
            )
        else:
            all_windows = {}
        all_windows = MPI.COMM_WORLD.bcast(all_windows, root=0)

        process_params = iteration.get_process_params()

        def process(observed_station, synthetic_station):
            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            # Make sure both have length 1.
            assert len(obs_tag) == 1, (
                "Station: %s - Requires 1 observed waveform tag. Has %i." % (
                    observed_station._station_name, len(obs_tag)))
            assert len(syn_tag) == 1, (
                "Station: %s - Requires 1 synthetic waveform tag. Has %i." % (
                    observed_station._station_name, len(syn_tag)))

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            # Finally get the data.
            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]

            # Extract coordinates once.
            coordinates = observed_station.coordinates

            # Process and rotate the synthetics.
            st_syn = self.comm.waveforms.rotate_and_process_synthetics(
                st=st_syn.copy(), station_id=observed_station._station_name,
                iteration=iteration, event_name=event["event_name"],
                coordinates=coordinates)

            adjoint_sources = {}

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)
                except LASIFNotFoundError:
                    continue

                if data_tr.id not in all_windows:
                    # No windows existing for components. Skipping.
                    continue

                # Collect all.
                adj_srcs = []

                windows = all_windows[data_tr.id]
                try:
                    for starttime, endtime in windows:
                        asrc = \
                            self.comm.wins_and_adj_sources.calculate_adjoint_source(
                            data=data_tr, synth=synth_tr, starttime=starttime,
                            endtime=endtime, taper="hann",
                            taper_percentage=0.05,
                            min_period=1.0/process_params["lowpass"],
                            max_period=1.0/process_params["highpass"],
                            ad_src_type="TimeFrequencyPhaseMisfitFichtner2008")
                        adj_srcs.append(asrc)
                except:
                    # Either pass or fail for the whole component.
                    continue

                if not adj_srcs:
                    continue

                # Sum up both misfit, and adjoint source.
                misfit = sum([_i["misfit_value"] for _i in adj_srcs])
                adj_source = np.sum([_i["adjoint_source"] for _i in adj_srcs],
                                    axis=0)

                adjoint_sources[data_tr.id] = {
                    "misfit": misfit,
                    "adj_source": adj_source
                }

            return adjoint_sources

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r")
        ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r")

        # Launch the processing. This will be executed in parallel across
        # ranks.
        results = ds.process_two_files_without_parallel_output(ds_synth,
                                                               process)

        # Write files on rank 0.
        if MPI.COMM_WORLD.rank == 0:
            self.comm.wins_and_adj_sources.write_adjoint_sources(
                event=event["event_name"], iteration=iteration,
                adj_sources=results)

    def select_windows(self, event, iteration_name, window_set_name, **kwargs):
        """
        Automatically select the windows for the given event and iteration.

        Function must be called with MPI.

        :param event: The event.
        :param iteration: The iteration.
        """
        from lasif.utils import select_component_from_stream

        from mpi4py import MPI
        import pyasdf

        event = self.comm.events.get(event)

        # Get the ASDF filenames.
        processed_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="processed",
            tag_or_iteration=self.comm.waveforms.preprocessing_tag)
        synthetic_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event["event_name"],
            data_type="synthetic",
            tag_or_iteration=iteration_name)

        if not os.path.exists(processed_filename):
            msg = "File '%s' does not exists." % processed_filename
            raise LASIFNotFoundError(msg)

        if not os.path.exists(synthetic_filename):
            msg = "File '%s' does not exists." % synthetic_filename
            raise LASIFNotFoundError(msg)

        # Load project specific window selection function.
        select_windows = self.comm.project.get_project_function(
            "window_picking_function")

        process_params = self.comm.project.preprocessing_params
        minimum_period = process_params["highpass_period"]
        maximum_period = process_params["lowpass_period"]

        def process(observed_station, synthetic_station):
            obs_tag = observed_station.get_waveform_tags()
            syn_tag = synthetic_station.get_waveform_tags()

            # Make sure both have length 1.
            assert len(obs_tag) == 1, (
                "Station: %s - Requires 1 observed waveform tag. Has %i." % (
                    observed_station._station_name, len(obs_tag)))
            assert len(syn_tag) == 1, (
                "Station: %s - Requires 1 synthetic waveform tag. Has %i." % (
                    observed_station._station_name, len(syn_tag)))

            obs_tag = obs_tag[0]
            syn_tag = syn_tag[0]

            # Finally get the data.
            st_obs = observed_station[obs_tag]
            st_syn = synthetic_station[syn_tag]

            # Extract coordinates once.
            coordinates=observed_station.coordinates

            #Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(), event_name=event["event_name"])

            all_windows = {}

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)

                    # Scale Data to synthetics
                    if True:
                        scaling_factor = synth_tr.data.ptp() / data_tr.data.ptp()
                        # Store and apply the scaling.
                        data_tr.stats.scaling_factor = scaling_factor
                        data_tr.data *= scaling_factor

                except LASIFNotFoundError:
                    continue

                windows = select_windows(
                    data_tr, synth_tr, event["latitude"],
                    event["longitude"], event["depth_in_km"],
                    coordinates["latitude"],
                    coordinates["longitude"],
                    minimum_period=minimum_period,
                    maximum_period=maximum_period,
                    iteration=iteration_name, **kwargs)

                if not windows:
                    continue
                all_windows[data_tr.id] = windows

            if all_windows:
                return all_windows

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r")
        ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r")

        # Launch the processing. This will be executed in parallel across
        # ranks.
        results = ds.process_two_files_without_parallel_output(ds_synth, process)

        # Write files on rank 0.
        if MPI.COMM_WORLD.rank == 0:
            print("Selected windows: ", results)
            self.comm.wins_and_adj_sources.write_windows_to_sql(event_name=event["event_name"], windows=results,
                                                                window_set_name=window_set_name)

    def select_windows_for_station(self, event, iteration, station, window_set_name, **kwargs):
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
        data = self.comm.query.get_matching_waveforms(event["event_name"], iteration,
                                                      station)

        process_params = self.comm.project.preprocessing_params
        minimum_period = process_params["highpass_period"]
        maximum_period = process_params["lowpass_period"]

        window_group_manager = self.comm.wins_and_adj_sources.get(window_set_name)

        # Delete the windows for this stations.
        print(station)
        #window_group_manager.(station)

        found_something = False
        for component in ["E", "N", "Z"]:
            try:
                data_tr = select_component_from_stream(data.data, component)
                synth_tr = select_component_from_stream(data.synthetics,
                                                        component)
                # delete preexisting windows
                window_group_manager.del_all_windows_from_event_channel(event["event_name"], data_tr.id)
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

            for starttime, endtime in windows:
                window_group_manager.add_window_to_event_channel(event_name=event["event_name"],
                                                                 channel_name=data_tr.id,
                                                                 start_time=starttime, end_time=endtime)

        if found_something is False:
            raise LASIFNotFoundError(
                "No matching data found for event '%s', iteration '%s', and "
                "station '%s'." % (event["event_name"], iteration.name,
                                   station))

    def generate_input_files(self, weight_set_name, iteration_name, event_name,
                             simulation_type):
        """
        Generate the input files for one event.

        :param iteration_name: The name of the iteration.
        :param event_name: The name of the event for which to generate the
            input files.
        :param simulation_type: The type of simulation to perform. Possible
            values are: 'normal simulate', 'adjoint forward', 'adjoint
            reverse'
        """

        # =====================================================================
        # read weights toml file, get event and list of stations
        # =====================================================================
        weights = self.comm.weights.get(weight_set_name)

        # Check that the event is part of the iterations.
        if event_name not in weights.events:
            msg = ("Event '%s' not part of iteration '%s'.\nEvents available "
                   "in iteration:\n\t%s" %
                   (event_name, weight_set_name, "\n\t".join(
                       sorted(weights.events.keys()))))
            raise ValueError(msg)

        asdf_file = self.comm.waveforms.get_asdf_filename(event_name=event_name, data_type="raw" )

        import pyasdf
        ds = pyasdf.ASDFDataSet(asdf_file)
        event = ds.events[0]

        # Build inventory of all stations present in ASDF file
        stations = ds.waveforms.list()
        inv = ds.waveforms[stations[0]].StationXML
        for station in stations[1:]:
            inv += ds.waveforms[station].StationXML

        import salvus_seismo
        src = salvus_seismo.Source.parse(event, sliprate="delta")
        recs = salvus_seismo.Receiver.parse(inv)

        solver_settings = self.comm.project.solver['settings']
        # Choose a center frequency suitable for our mesh.
        src.center_frequency = solver_settings['simulation_parameters']['source_center_frequency']
        mesh_file = self.comm.project.config['mesh_file']

        # Generate the configuration object for salvus_seismo
        config = salvus_seismo.Config(
            mesh_file=mesh_file,
            end_time=solver_settings['simulation_parameters']['end_time'],
            salvus_call=solver_settings['computational_setup']['salvus_call'],
            polynomial_order=solver_settings['simulation_parameters']['polynomial_order'],
            verbose=True,
            dimensions=solver_settings['simulation_parameters']['dimensions'])

        # =================================================================
        # output
        # =================================================================

        long_iter_name = self.comm.iterations.get_long_iteration_name(iteration_name)
        input_files_dir = self.comm.project.paths['salvus_input']
        output_dir = os.path.join(input_files_dir, long_iter_name, event_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        import shutil
        shutil.rmtree(output_dir)
        salvus_seismo.generate_cli_call(
            source=src, receivers=recs, config=config,
            output_folder=output_dir,
            exodus_file=mesh_file)

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
            print(".",)
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

        print("Wrote adjoint sources for %i station(s) to %s." % (
            len(adjoint_source_stations), os.path.relpath(output_folder)))
