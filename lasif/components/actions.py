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

    def process_data(self, events):
        """
        Preprocesses all data for a given iteration.

        This function works with and without MPI.

        :param event_names: event_ids is a list of events to process in this
            run. It will process all events if not given.
        """
        from mpi4py import MPI
        process_params = self.comm.project.processing_params
        simulation_params = self.comm.project.simulation_params
        npts = simulation_params["number_of_time_steps"]
        dt = simulation_params["time_increment"]
        salvus_start_time = simulation_params["start_time"]

        def processing_data_generator():
            """
            Generate a dictionary with information for processing for each
            waveform.
            """
            # Loop over the chosen events.
            for event_name in events:
                output_folder = os.path.join(
                    self.comm.project.paths["preproc_eq_data"], event_name)

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                asdf_file_name = self.comm.waveforms.get_asdf_filename(
                    event_name, data_type="raw")
                lowpass_period = process_params["lowpass_period"]
                highpass_period = process_params["highpass_period"]
                preprocessing_tag = self.comm.waveforms.preprocessing_tag

                output_filename = os.path.join(output_folder,
                                               preprocessing_tag + ".h5")

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

    def calculate_adjoint_sources(self, event, iteration, window_set_name,
                                  **kwargs):
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
            tag_or_iteration=iteration)

        if not os.path.exists(processed_filename):
            msg = "File '%s' does not exists." % processed_filename
            raise LASIFNotFoundError(msg)

        if not os.path.exists(synthetic_filename):
            msg = "File '%s' does not exists." % synthetic_filename
            raise LASIFNotFoundError(msg)

        # Read all windows on rank 0 and broadcast.
        if MPI.COMM_WORLD.rank == 0:
            all_windows = self.comm.wins_and_adj_sources.read_all_windows(
                event=event["event_name"], window_set_name=window_set_name
            )
        else:
            all_windows = {}
        all_windows = MPI.COMM_WORLD.bcast(all_windows, root=0)

        process_params = self.comm.project.processing_params

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

            # Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(), event_name=event["event_name"])

            adjoint_sources = {}

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)

                except LASIFNotFoundError:
                    continue

                if self.comm.project.processing_params["scale_data_"
                                                       "to_synthetics"]:
                    scaling_factor = synth_tr.data.ptp() / data_tr.data.ptp()
                    # Store and apply the scaling.
                    data_tr.stats.scaling_factor = scaling_factor
                    data_tr.data *= scaling_factor

                net, sta, cha = data_tr.id.split(".", 2)
                station = net + "." + sta

                if station not in all_windows:
                    continue
                if data_tr.id not in all_windows[station]:
                    continue

                # Collect all.
                adj_srcs = []
                windows = all_windows[station][data_tr.id]
                try:
                    for starttime, endtime in windows:
                        asrc = \
                            self.comm.\
                            wins_and_adj_sources.calculate_adjoint_source(
                                data=data_tr, synth=synth_tr,
                                starttime=starttime, endtime=endtime,
                                taper="hann", taper_percentage=0.05,
                                min_period=process_params["highpass_period"],
                                max_period=process_params["lowpass_period"],
                                ad_src_type="TimeFrequency"
                                            "PhaseMisfitFichtner2008")
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

        process_params = self.comm.project.processing_params
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
            coordinates = observed_station.coordinates

            # Process the synthetics.
            st_syn = self.comm.waveforms.process_synthetics(
                st=st_syn.copy(), event_name=event["event_name"])

            all_windows = {}

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)

                    if self.comm.project.processing_params["scale_data_"
                                                           "to_synthetics"]:
                        scaling_factor = \
                            synth_tr.data.ptp() / data_tr.data.ptp()
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
        results = ds.process_two_files_without_parallel_output(
            ds_synth, process)

        # Write files on rank 0.
        if MPI.COMM_WORLD.rank == 0:
            print("Selected windows: ", results)
            self.comm.wins_and_adj_sources.write_windows_to_sql(
                event_name=event["event_name"], windows=results,
                window_set_name=window_set_name)

    def select_windows_for_station(self, event, iteration, station,
                                   window_set_name, **kwargs):
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
        data = self.comm.query.get_matching_waveforms(event["event_name"],
                                                      iteration, station)

        process_params = self.comm.project.processing_params
        minimum_period = process_params["highpass_period"]
        maximum_period = process_params["lowpass_period"]

        window_group_manager = self.comm.wins_and_adj_sources.get(
            window_set_name)

        found_something = False
        for component in ["E", "N", "Z"]:
            try:
                data_tr = select_component_from_stream(data.data, component)
                synth_tr = select_component_from_stream(data.synthetics,
                                                        component)
                # delete preexisting windows
                window_group_manager.del_all_windows_from_event_channel(
                    event["event_name"], data_tr.id)
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
                window_group_manager.add_window_to_event_channel(
                    event_name=event["event_name"],
                    channel_name=data_tr.id,
                    start_time=starttime, end_time=endtime)

        if found_something is False:
            raise LASIFNotFoundError(
                "No matching data found for event '%s', iteration '%s', and "
                "station '%s'." % (event["event_name"], iteration.name,
                                   station))

    def generate_input_files(self, iteration_name, event_name,
                             simulation_type="forward"):
        """
        Generate the input files for one event.

        :param iteration_name: The name of the iteration.
        :param event_name: The name of the event for which to generate the
            input files.
        """

        # =====================================================================
        # read weights toml file, get event and list of stations
        # =====================================================================
        asdf_file = self.comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="raw")

        import pyasdf
        ds = pyasdf.ASDFDataSet(asdf_file)
        event = ds.events[0]

        # Build inventory of all stations present in ASDF file
        stations = ds.waveforms.list()
        inv = ds.waveforms[stations[0]].StationXML
        for station in stations[1:]:
            inv += ds.waveforms[station].StationXML

        import salvus_seismo

        src_time_func = self.comm.project.\
            computational_setup["source_time_function_type"]

        if src_time_func == "bandpass_filtered_heaviside":
            salvus_seismo_src_time_func = "heaviside"
        else:
            salvus_seismo_src_time_func = src_time_func

        src = salvus_seismo.Source.parse(
            event,
            sliprate=salvus_seismo_src_time_func)
        recs = salvus_seismo.Receiver.parse(inv)

        simulation_parameters = self.comm.project.simulation_params
        mesh_file = self.comm.project.config["mesh_file"]

        # Generate the configuration object for salvus_seismo
        if simulation_type == "forward":
            config = salvus_seismo.Config(
                mesh_file=mesh_file,
                start_time=simulation_parameters["start_time"],
                time_step=simulation_parameters["time_increment"],
                end_time=simulation_parameters["end_time"],
                salvus_call=self.comm.project.
                    computational_setup["salvus_call"],
                polynomial_order=simulation_parameters["polynomial_order"],
                verbose=True,
                dimensions=3,
                num_absorbing_layers=
                simulation_parameters["number_of_absorbing_layers"],
                with_anisotropy=self.comm.project.
                computational_setup["with_anisotropy"],
                wavefield_file_name="wavefield.h5",
                wavefield_fields="adjoint")

        elif simulation_type == "step_length":
            config = salvus_seismo.Config(
                mesh_file=mesh_file,
                start_time=simulation_parameters["start_time"],
                time_step=simulation_parameters["time_increment"],
                end_time=simulation_parameters["end_time"],
                salvus_call=self.comm.project.
                    computational_setup["salvus_call"],
                polynomial_order=simulation_parameters["polynomial_order"],
                verbose=True,
                dimensions=3,
                num_absorbing_layers=
                simulation_parameters["number_of_absorbing_layers"],
                with_anisotropy=self.comm.project.
                computational_setup["with_anisotropy"])

        # ==============================================================j===
        # output
        # =================================================================
        long_iter_name = self.comm.iterations.get_long_iteration_name(
            iteration_name)
        input_files_dir = self.comm.project.paths['salvus_input']
        output_dir = os.path.join(input_files_dir, long_iter_name, event_name,
                                  simulation_type)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        import shutil
        shutil.rmtree(output_dir)
        salvus_seismo.generate_cli_call(
            source=src, receivers=recs, config=config,
            output_folder=output_dir,
            exodus_file=mesh_file)

        if src_time_func == "bandpass_filtered_heaviside":
            self.write_custom_stf(output_dir)

        run_salvus = os.path.join(output_dir, "run_salvus.sh")
        if simulation_type == "forward":
            with open(run_salvus, "a") as fh:
                fh.write(" --io-sampling-rate-volume 10"
                         " --io-memory-per-rank-in-MB 5000")

    def write_custom_stf(self, output_dir):
        import toml
        import h5py

        source_toml = os.path.join(output_dir, "source.toml")
        with open(source_toml, "r") as fh:
            source_dict = toml.load(fh)['source'][0]

        location = source_dict['location']
        moment_tensor = source_dict['scale']

        freqmax = 1.0 / self.comm.project.processing_params["highpass_period"]
        freqmin = 1.0 / self.comm.project.processing_params["lowpass_period"]

        delta = self.comm.project.simulation_params["time_increment"]
        npts = self.comm.project.simulation_params["number_of_time_steps"]

        stf_fct = self.comm.project.get_project_function(
            "source_time_function")

        stf = stf_fct(npts=npts, delta=delta,
                              freqmin=freqmin, freqmax=freqmax)

        stf_mat = np.zeros((len(stf), len(moment_tensor)))
        for i, moment in enumerate(moment_tensor):
            stf_mat[:, i] = stf * moment

        heaviside_file_name = os.path.join(output_dir, "Heaviside.h5")
        f = h5py.File(heaviside_file_name, 'w')

        source = f.create_dataset("source", data=stf_mat)
        source.attrs["dt"] = delta
        source.attrs["location"] = location
        source.attrs["spatial-type"] = np.string_("moment_tensor")
        source.attrs["starttime"] = -delta

        f.close()

        # remove source toml and write new one
        os.remove(source_toml)
        source_str = f"source_input_file = \"{heaviside_file_name}\"\n\n" \
                     f"[[source]]\n" \
                     f"name = \"source\"\n" \
                     f"dataset_name = \"/source\""

        with open(source_toml, "w") as fh:
            fh.write(source_str)

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
        import pyasdf
        import toml
        import h5py
        import shutil

        # This will do stuff for each event and a single iteration

        # Step one, read adj_src file that should have been created already
        event = self.comm.events.get(event_name)
        iteration = self.comm.iterations.get_long_iteration_name(iteration_name)

        adj_src_file = self.comm.wins_and_adj_sources.get_filename(event, iteration)

        ds = pyasdf.ASDFDataSet(adj_src_file)
        adj_srcs = ds.auxiliary_data["AdjointSources"]

        # Load receiver toml file
        long_iter_name = self.comm.iterations.get_long_iteration_name(
            iteration_name)
        input_files_dir = self.comm.project.paths['salvus_input']
        receiver_dir = os.path.join(input_files_dir, long_iter_name, event_name,
                                  "forward")
        output_dir = os.path.join(input_files_dir, long_iter_name, event_name,
                                  "adjoint")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        receivers = toml.load(os.path.join(receiver_dir, "receivers.toml"))["receiver"]

        adjoint_source_file_name = os.path.join(output_dir, "adjoint_source.h5")
        toml_file_name = os.path.join(output_dir, "adjoint.toml")

        toml_string = "source_input_file = \"{}\"\n\n".format(adjoint_source_file_name)

        f = h5py.File(adjoint_source_file_name, 'w')


        for adj_src in adj_srcs:
            station_name = adj_src.auxiliary_data_type.split("/")[1]
            channels = adj_src.list()

            e_comp = np.zeros_like(adj_src[channels[0]].data.value)
            n_comp = np.zeros_like(adj_src[channels[0]].data.value)
            z_comp = np.zeros_like(adj_src[channels[0]].data.value)

            for channel in channels:
                # check channel and set component
                if channel[-1] == "E":
                    e_comp = adj_src[channel].data.value
                elif channel[-1] == "N":
                    n_comp = adj_src[channel].data.value
                elif channel[-1] == "Z":
                    z_comp = adj_src[channel].data.value
                print(np.sum(z_comp))
                zne = np.array((z_comp, n_comp, e_comp)).T

            for receiver in receivers:

                station = receiver["network"] + "_" + receiver["station"]


                if station == station_name:
                    print("writing source")
                    transform_mat = np.array(receiver["transform_matrix"])
                    xyz = np.dot(zne, transform_mat.T)

                    source = f.create_dataset(station, data=xyz)
                    source.attrs["dt"] = self.comm.project.simulation_params["time_increment"]
                    source.attrs['location'] = np.array(receiver["salvus_coordinates"])
                    source.attrs['spatial-type'] = np.string_("vector")
                    source.attrs['starttime'] = self.comm.project.simulation_params["start_time"]

                    toml_string += f"[[source]]\n" \
                                   f"name = \"{station}\"\n" \
                                   f"dataset_name = \"/{station}\"\n\n"

        f.close()
        with open(toml_file_name, "w") as fh:
            fh.write(toml_string)

        mesh_file = self.comm.project.config["mesh_file"]
        simulation_params = self.comm.project.simulation_params
        start_time = simulation_params["start_time"]
        end_time = simulation_params["end_time"]
        time_step = simulation_params["time_increment"]
        num_absorbing_layers = simulation_params["number_of_absorbing_layers"]
        polynomial_order = simulation_params["polynomial_order"]

        possible_boundaries = set(("r0", "t0", "t1", "p0", "p1",
                                   "inner_boundary"))
        absorbing_boundaries = \
            possible_boundaries.intersection(set(self.comm.project.domain.get_side_set_names()))
        if absorbing_boundaries:
            absorbing_boundaries = ",".join(sorted(absorbing_boundaries))
            print("Automatically determined the following absorbing "
                  "boundary side sets: %s" % absorbing_boundaries)


        salvus_command = \
            f"mpirun -n 4 --dimension 3 --mesh-file {mesh_file} " \
            f"--model-file {mesh_file} --start-time {start_time} " \
            f"--time-step {time_step} --num-absorbing-layers {num_absorbing_layers} " \
            f"--end-time {end_time} --polynomial-order {polynomial_order} " \
            f"--adjoint --kernel-file kernel_{event_name}.e --load-fields adjoint " \
            f"--load-wavefield-file wavefield.h5 --save-static-fields gradient " \
            f"--save-static-file-name {event_name}.h5 --kernel-fields TTI " \
            f"--io-memory-per-rank-in-MB 5000 --with-anisotropy " \
            f"--absorbing-boundaries {absorbing_boundaries} " \
            f"--source-toml {toml_file_name}"
        salvus_command_file = os.path.join(output_dir, "run_salvus.sh")
        with open(salvus_command_file, "w") as fh:
            fh.write(salvus_command)
