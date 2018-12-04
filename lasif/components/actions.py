#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import os
import pathlib
import typing

from lasif import LASIFError, LASIFNotFoundError

from .component import Component
from mpi4py import MPI


def process_two_files_without_parallel_output(ds, other_ds,
                                              process_function,
                                              traceback_limit=3):
    import traceback
    import sys
    """
    Process data in two data sets.

    This is mostly useful for comparing data in two data sets in any
    number of scenarios. It again takes a function and will apply it on
    each station that is common in both data sets. Please see the
    :doc:`parallel_processing` document for more details.

    Can only be run with MPI.

    :type other_ds: :class:`.ASDFDataSet`
    :param other_ds: The data set to compare to.
    :param process_function: The processing function takes two
        parameters: The station group from this data set and
        the matching station group from the other data set.
    :type traceback_limit: int
    :param traceback_limit: The length of the traceback printed if an
        error occurs in one of the workers.
    :return: A dictionary for each station with gathered values. Will
        only be available on rank 0.
    """

    # Collect the work that needs to be done on rank 0.
    if MPI.COMM_WORLD.rank == 0:

        def split(container, count):
            """
            Simple function splitting a container into equal length
            chunks.
            Order is not preserved but this is potentially an advantage
            depending on the use case.
            """
            return [container[_i::count] for _i in range(count)]

        this_stations = set(ds.waveforms.list())
        other_stations = set(other_ds.waveforms.list())

        # Usable stations are those that are part of both.
        usable_stations = list(
            this_stations.intersection(other_stations))
        total_job_count = len(usable_stations)
        jobs = split(usable_stations, MPI.COMM_WORLD.size)
    else:
        jobs = None

    # Scatter jobs.
    jobs = MPI.COMM_WORLD.scatter(jobs, root=0)

    # Dictionary collecting results.
    results = {}

    for _i, station in enumerate(jobs):

        if MPI.COMM_WORLD.rank == 0:
            print(" -> Processing approximately task %i of %i ..." %
                  ((_i * MPI.COMM_WORLD.size + 1), total_job_count),
                  flush=True)
        try:
            result = process_function(
                getattr(ds.waveforms, station),
                getattr(other_ds.waveforms, station))
            # print("Working", flush=True)
        except Exception:
            # print("Not working", flush=True)
            # If an exception is raised print a good error message
            # and traceback to help diagnose the issue.
            msg = ("\nError during the processing of station '%s' "
                   "on rank %i:" % (station, MPI.COMM_WORLD.rank))

            # Extract traceback from the exception.
            exc_info = sys.exc_info()
            stack = traceback.extract_stack(
                limit=traceback_limit)
            tb = traceback.extract_tb(exc_info[2])
            full_tb = stack[:-1] + tb
            exc_line = traceback.format_exception_only(
                *exc_info[:2])
            tb = ("Traceback (At max %i levels - most recent call "
                  "last):\n" % traceback_limit)
            tb += "".join(traceback.format_list(full_tb))
            tb += "\n"
            # A bit convoluted but compatible with Python 2 and
            # 3 and hopefully all encoding problems.
            tb += "".join(
                _i.decode(errors="ignore")
                if hasattr(_i, "decode") else _i
                for _i in exc_line)

            # These potentially keep references to the HDF5 file
            # which in some obscure way and likely due to
            # interference with internal HDF5 and Python references
            # prevents it from getting garbage collected. We
            # explicitly delete them here and MPI can finalize
            # afterwards.
            del exc_info
            del stack

            print(msg, flush=True)
            print(tb, flush=True)
        else:
            # print("Else!", flush=True)
            results[station] = result
    # barrier but better be safe than sorry.
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.rank == 0:
        print("All ranks finished", flush=True)
    return results


class ActionsComponent(Component):
    """
    Component implementing actions on the data. Requires most other
    components to be available.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """

    def process_data(self, events):
        """
        Processes all data for a given iteration.

        This function works with and without MPI.

        :param events: event_ids is a list of events to process in this
            run. It will process all events if not given.
        """
        from mpi4py import MPI
        process_params = self.comm.project.processing_params
        solver_settings = self.comm.project.solver_settings
        npts = solver_settings["number_of_time_steps"]
        dt = solver_settings["time_increment"]
        salvus_start_time = solver_settings["start_time"]

        def processing_data_generator():
            """
            Generate a dictionary with information for processing for each
            waveform.
            """
            # Loop over the chosen events.
            for event_name in events:
                output_folder = os.path.join(
                    self.comm.project.paths["preproc_eq_data"], event_name)
                asdf_file_name = self.comm.waveforms.get_asdf_filename(
                    event_name, data_type="raw")
                preprocessing_tag = self.comm.waveforms.preprocessing_tag
                output_filename = os.path.join(output_folder,
                                               preprocessing_tag + ".h5")

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                lowpass_period = process_params["lowpass_period"]
                highpass_period = process_params["highpass_period"]

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
                    "highpass_period": highpass_period,
                }
                yield ret_dict

        to_be_processed = [{"processing_info": _i}
                           for _i in processing_data_generator()]

        # Load project specific data processing function.
        preprocessing_function_asdf = self.comm.project.get_project_function(
            "preprocessing_function_asdf")
        MPI.COMM_WORLD.Barrier()
        for event in to_be_processed:
            preprocessing_function_asdf(event["processing_info"])
            MPI.COMM_WORLD.Barrier()

    def calculate_adjoint_sources(self, event, iteration, window_set_name,
                                  plot=False, **kwargs):
        from lasif.utils import select_component_from_stream

        from mpi4py import MPI
        import pyasdf
        import salvus_misfit

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
            all_windows = self.comm.windows.read_all_windows(
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
                st=st_syn.copy(), event_name=event["event_name"],
                iteration=iteration)

            adjoint_sources = {}
            ad_src_type = self.comm.project.config["misfit_type"]
            if ad_src_type == "weighted_waveform_misfit":
                env_scaling = True
                ad_src_type = "waveform_misfit"
            else:
                env_scaling = False

            for component in ["E", "N", "Z"]:
                try:
                    data_tr = select_component_from_stream(st_obs, component)
                    synth_tr = select_component_from_stream(st_syn, component)
                except LASIFNotFoundError:
                    continue

                if self.comm.project.processing_params["scale_data_"
                                                       "to_synthetics"]:
                    if not self.comm.project.config["misfit_type"] == \
                            "L2NormWeighted":
                        scaling_factor = \
                            synth_tr.data.ptp() / data_tr.data.ptp()
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
                windows = all_windows[station][data_tr.id]
                try:
                    # for window in windows:
                    asrc = salvus_misfit.calculate_adjoint_source(
                        observed=data_tr, synthetic=synth_tr,
                        window=windows,
                        min_period=process_params["highpass_period"],
                        max_period=process_params["lowpass_period"],
                        adj_src_type=ad_src_type,
                        window_set=window_set_name,
                        taper_ratio=0.15, taper_type='cosine',
                        plot=plot, envelope_scaling=env_scaling)
                except:
                    # Either pass or fail for the whole component.
                    continue

                if not asrc:
                    continue
                # Sum up both misfit, and adjoint source.
                misfit = asrc.misfit
                adj_source = asrc.adjoint_source
                # Time reversal is currently needed in Salvus but that will
                # change later and this can be removed
                adj_source = adj_source[::-1]

                adjoint_sources[data_tr.id] = {
                    "misfit": misfit,
                    "adj_source": adj_source
                }

            return adjoint_sources

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r")
        ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r")

        # Launch the processing. This will be executed in parallel across
        # ranks.
        results = process_two_files_without_parallel_output(ds, ds_synth,
                                                            process)

        # Write files on all ranks.
        filename = self.comm.adj_sources.get_filename(
            event=event["event_name"], iteration=iteration)
        ad_src_counter = 0
        size = MPI.COMM_WORLD.size
        MPI.COMM_WORLD.Barrier()
        for thread in range(size):
            rank = MPI.COMM_WORLD.rank
            if rank == thread:
                print(
                    f"Writing adjoint sources for rank: {rank+1} "
                    f"out of {size}", flush=True)
                with pyasdf.ASDFDataSet(filename=filename, mpi=False,
                                        mode="a") as bs:
                    for value in results.values():
                        if not value:
                            continue
                        for c_id, adj_source in value.items():
                            net, sta, loc, cha = c_id.split(".")
                            bs.add_auxiliary_data(
                                data=adj_source["adj_source"],
                                data_type="AdjointSources",
                                path="%s_%s/Channel_%s_%s" % (net, sta,
                                                              loc, cha),
                                parameters={"misfit": adj_source["misfit"]})
                        ad_src_counter += 1

            MPI.COMM_WORLD.barrier()
        if MPI.COMM_WORLD.rank == 0:
            with pyasdf.ASDFDataSet(filename=filename, mpi=False,
                                    mode="a")as ds:
                length = len(ds.auxiliary_data.AdjointSources.list())
            print(f"{length} Adjoint sources are in your file.")

    def select_windows(self, event, iteration_name, window_set_name, **kwargs):
        """
        Automatically select the windows for the given event and iteration.

        Function must be called with MPI.

        :param event: The event.
        :param iteration_name: The iteration.
        :param window_set_name: The name of the window set to pick into
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

        # Get source time function
        stf_fct = self.comm.project.get_project_function(
            "source_time_function")
        delta = self.comm.project.solver_settings["time_increment"]
        npts = self.comm.project.solver_settings["number_of_time_steps"]
        freqmax = 1.0 / self.comm.project.processing_params["highpass_period"]
        freqmin = 1.0 / self.comm.project.processing_params["lowpass_period"]
        stf_trace = stf_fct(npts=npts, delta=delta, freqmin=freqmin,
                            freqmax=freqmax)

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
                st=st_syn.copy(),
                event_name=event["event_name"], iteration=iteration_name)

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

                windows = None
                try:
                    windows = select_windows(
                        data_tr, synth_tr, stf_trace, event["latitude"],
                        event["longitude"], event["depth_in_km"],
                        coordinates["latitude"],
                        coordinates["longitude"],
                        minimum_period=minimum_period,
                        maximum_period=maximum_period,
                        iteration=iteration_name, **kwargs)
                except Exception as e:
                    print(e)

                if not windows:
                    continue
                all_windows[data_tr.id] = windows

            if all_windows:
                return all_windows

        ds = pyasdf.ASDFDataSet(processed_filename, mode="r")
        ds_synth = pyasdf.ASDFDataSet(synthetic_filename, mode="r")

        results = process_two_files_without_parallel_output(ds, ds_synth,
                                                            process)
        MPI.COMM_WORLD.Barrier()
        # Write files on rank 0.
        if MPI.COMM_WORLD.rank == 0:
            print("Finished window selection", flush=True)
        size = MPI.COMM_WORLD.size
        MPI.COMM_WORLD.Barrier()
        for thread in range(size):
            rank = MPI.COMM_WORLD.rank
            if rank == thread:
                print(
                    f"Writing windows for rank: {rank+1} "
                    f"out of {size}", flush=True)
                self.comm.windows.write_windows_to_sql(
                    event_name=event["event_name"], windows=results,
                    window_set_name=window_set_name)
            MPI.COMM_WORLD.Barrier()

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

        # Get source time function
        stf_fct = self.comm.project.get_project_function(
            "source_time_function")
        delta = self.comm.project.solver_settings["time_increment"]
        npts = self.comm.project.solver_settings["number_of_time_steps"]
        freqmax = 1.0 / self.comm.project.processing_params["highpass_period"]
        freqmin = 1.0 / self.comm.project.processing_params["lowpass_period"]
        stf_trace = stf_fct(npts=npts, delta=delta, freqmin=freqmin,
                            freqmax=freqmax)

        process_params = self.comm.project.processing_params
        minimum_period = process_params["highpass_period"]
        maximum_period = process_params["lowpass_period"]

        window_group_manager = self.comm.windows.get(
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

            windows = select_windows(data_tr, synth_tr, stf_trace,
                                     event["latitude"],
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
                             simulation_type="forward", previous_iteration=None):
        """
        Generate the input files for one event.

        :param iteration_name: The name of the iteration.
        :param event_name: The name of the event for which to generate the
            input files.
        :param simulation_type: forward, adjoint, step_length
        :param previous_iteration: name of the iteration to copy input files
            from.
        """
        import shutil
        if self.comm.project.config["mesh_file"] == "multiple":
            mesh_file = os.path.join(self.comm.project.paths["models"],
                                     "EVENT_SPECIFIC", event_name, "mesh.e")
        else:
            mesh_file = self.comm.project.config["mesh_file"]

        input_files_dir = self.comm.project.paths['salvus_input']

        # If previous iteration specified, copy files over and update mesh_file
        # This part could be extended such that other parameters can be
        # updated as well.
        if previous_iteration:
            long_prev_iter_name = self.comm.iterations.get_long_iteration_name(
                previous_iteration)
            prev_it_dir = os.path.join(input_files_dir, long_prev_iter_name,
                                       event_name, simulation_type)
        if previous_iteration and os.path.exists(prev_it_dir):
            long_iter_name = self.comm.iterations.get_long_iteration_name(
                iteration_name)
            output_dir = os.path.join(input_files_dir, long_iter_name,
                                      event_name,
                                      simulation_type)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not prev_it_dir == output_dir:
                shutil.copyfile(os.path.join(prev_it_dir, "run_salvus.sh"),
                                os.path.join(output_dir, "run_salvus.sh"))
            else:
                print("Previous iteration is identical to current iteration.")
            with open(os.path.join(output_dir, "run_salvus.sh"), "r") as fh:
                cmd_string = fh.read()
            l = cmd_string.split(" ")
            l[l.index("--model-file") + 1] = mesh_file
            l[l.index("--mesh-file") + 1] = mesh_file
            cmd_string = " ".join(l)
            with open(os.path.join(output_dir, "run_salvus.sh"), "w") as fh:
                fh.write(cmd_string)
            return
        elif previous_iteration and not os.path.exists(prev_it_dir):
            print(f"Could not find previous iteration directory for event: "
                  f"{event_name}, generating input files")

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

        src_time_func = self.comm.project. \
            solver_settings["source_time_function_type"]

        if src_time_func == "bandpass_filtered_heaviside":
            salvus_seismo_src_time_func = "heaviside"
        else:
            salvus_seismo_src_time_func = src_time_func

        src = salvus_seismo.Source.parse(
            event,
            sliprate=salvus_seismo_src_time_func)
        recs = salvus_seismo.Receiver.parse(inv)

        solver_settings = self.comm.project.solver_settings
        if solver_settings["number_of_absorbing_layers"] == 0:
            num_absorbing_layers = None
        else:
            num_absorbing_layers = \
                solver_settings["number_of_absorbing_layers"]

        # Generate the configuration object for salvus_seismo
        if simulation_type == "forward":
            config = salvus_seismo.Config(
                mesh_file=mesh_file,
                start_time=solver_settings["start_time"],
                time_step=solver_settings["time_increment"],
                end_time=solver_settings["end_time"],
                salvus_call=self.comm.project.
                solver_settings["salvus_call"],
                polynomial_order=solver_settings["polynomial_order"],
                verbose=True,
                dimensions=3,
                num_absorbing_layers=num_absorbing_layers,
                with_anisotropy=self.comm.project.
                solver_settings["with_anisotropy"],
                wavefield_file_name="wavefield.h5",
                wavefield_fields="adjoint")

        elif simulation_type == "step_length":
            config = salvus_seismo.Config(
                mesh_file=mesh_file,
                start_time=solver_settings["start_time"],
                time_step=solver_settings["time_increment"],
                end_time=solver_settings["end_time"],
                salvus_call=self.comm.project.
                solver_settings["salvus_call"],
                polynomial_order=solver_settings["polynomial_order"],
                verbose=True,
                dimensions=3,
                num_absorbing_layers=num_absorbing_layers,
                with_anisotropy=self.comm.project.
                solver_settings["with_anisotropy"])

        # =====================================================================
        # output
        # =====================================================================
        long_iter_name = self.comm.iterations.get_long_iteration_name(
            iteration_name)

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

        self.write_custom_stf(output_dir)

        run_salvus = os.path.join(output_dir, "run_salvus.sh")
        io_sampling_rate = self.comm.project. \
            solver_settings["io_sampling_rate_volume"]
        memory_per_rank = self.comm.project.\
            solver_settings["io_memory_per_rank_in_MB"]
        if self.comm.project.solver_settings["with_attenuation"]:
            with open(run_salvus, "a") as fh:
                fh.write(f" --with-attenuation")
        if simulation_type == "forward":
            with open(run_salvus, "a") as fh:
                fh.write(f" --io-sampling-rate-volume {io_sampling_rate}"
                         f" --io-memory-per-rank-in-MB {memory_per_rank}"
                         f" --io-file-format bin")

    def write_custom_stf(self, output_dir):
        import toml

        source_toml = os.path.join(output_dir, "source.toml")
        with open(source_toml, "r") as fh:
            source_dict = toml.load(fh)['source'][0]

        heaviside_file_name = os.path.join(output_dir, "Heaviside.h5")

        new_source_dict = self._write_custom_stf(
            source_dict=source_dict, output_filename=heaviside_file_name)

        # Overwrite the source.toml file.
        with open(source_toml, "w") as fh:
            toml.dump(new_source_dict, fh)

    def _write_custom_stf(self, source_dict: typing.Dict,
                          output_filename: pathlib.Path) -> typing.Dict:
        """
        Writes a custom STF HDF5 file to the output_filename.

        Returns information that could be written to a new source TOML file.
        """
        import h5py

        location = source_dict['location']
        moment_tensor = source_dict['scale']

        freqmax = 1.0 / self.comm.project.processing_params["highpass_period"]
        freqmin = 1.0 / self.comm.project.processing_params["lowpass_period"]

        delta = self.comm.project.solver_settings["time_increment"]
        npts = self.comm.project.solver_settings["number_of_time_steps"]

        stf_fct = self.comm.project.get_project_function(
            "source_time_function")
        stf = self.comm.project.processing_params["stf"]
        if stf == "bandpass_filtered_heaviside":
            stf = stf_fct(npts=npts, delta=delta,
                          freqmin=freqmin, freqmax=freqmax)
        elif stf == "heaviside":
            stf = stf_fct(npts=npts, delta=delta)
        else:
            raise LASIFError(f"{stf} is not supported by lasif. Use either "
                             f"bandpass_filtered_heaviside or heaviside.")

        stf_mat = np.zeros((len(stf), len(moment_tensor)))
        for i, moment in enumerate(moment_tensor):
            stf_mat[:, i] = stf * moment

        with h5py.File(output_filename, 'w') as f:
            source = f.create_dataset("source", data=stf_mat)
            source.attrs["dt"] = delta
            source.attrs["location"] = location
            source.attrs["spatial-type"] = np.string_("moment_tensor")
            # Start time in nanoseconds
            source.attrs["starttime"] = -delta * 1.0e9

        return {"source_input_file": str(output_filename), "source": [
            {"name": "source", "dataset_name": "/source"}
        ]}

    def finalize_adjoint_sources(self, iteration_name, event_name,
                                 weight_set_name=None):
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
        iteration = self.comm.iterations.\
            get_long_iteration_name(iteration_name)

        adj_src_file = self.comm.adj_sources.\
            get_filename(event, iteration)

        ds = pyasdf.ASDFDataSet(adj_src_file)
        adj_srcs = ds.auxiliary_data["AdjointSources"]

        # Load receiver toml file
        long_iter_name = self.comm.iterations.get_long_iteration_name(
            iteration_name)
        input_files_dir = self.comm.project.paths['salvus_input']
        receiver_dir = os.path.join(input_files_dir, long_iter_name,
                                    event_name, "forward")
        with open(os.path.join(receiver_dir, "run_salvus.sh"), "r") as fh:
            cmd_string = fh.read()
        l = cmd_string.split(" ")
        receivers_file = l[l.index("--receiver-toml") + 1]

        output_dir = os.path.join(input_files_dir, long_iter_name,
                                  event_name, "adjoint")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        receivers = toml.load(
            os.path.join(receivers_file))["receiver"]

        adjoint_source_file_name = os.path.join(
            output_dir, "adjoint_source.h5")
        toml_file_name = os.path.join(output_dir, "adjoint.toml")

        toml_string = f"source_input_file = \"{adjoint_source_file_name}\"\n\n"
        f = h5py.File(adjoint_source_file_name, 'w')

        event_weight = 1.0
        if weight_set_name:
            ws = self.comm.weights.get(weight_set_name)
            event_weight = ws.events[event_name]["event_weight"]
            station_weights = ws.events[event_name]["stations"]

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
                zne = np.array((z_comp, n_comp, e_comp))
            for receiver in receivers:

                station = receiver["network"] + "_" + receiver["station"]

                if station == station_name:
                    print(f"writing adjoint source for station: {station}")
                    transform_mat = np.array(receiver["transform_matrix"])
                    xyz = np.dot(transform_mat.T, zne).T

                    net_dot_sta = \
                        receiver["network"] + "." + receiver["station"]
                    if weight_set_name:
                        weight = \
                            station_weights[net_dot_sta]["station_weight"] * \
                            event_weight
                        xyz *= weight

                    source = f.create_dataset(station, data=xyz)
                    source.attrs["dt"] = self.comm.project. \
                        solver_settings["time_increment"]
                    source.attrs['location'] = np.array(
                        receiver["salvus_coordinates"])
                    source.attrs['spatial-type'] = np.string_("vector")
                    # Start time in nanoseconds
                    source.attrs['starttime'] = self.comm.project. \
                        solver_settings["start_time"] * 1.0e9

                    toml_string += f"[[source]]\n" \
                                   f"name = \"{station}\"\n" \
                                   f"dataset_name = \"/{station}\"\n\n"

        f.close()
        with open(toml_file_name, "w") as fh:
            fh.write(toml_string)


        ########################################
        # Input file writing things below.
        ########################################
        if self.comm.project.config["mesh_file"] == "multiple":
            mesh_file = os.path.join(self.comm.project.paths["models"],
                                     "EVENT_SPECIFIC", event_name, "mesh.e")
        else:
            mesh_file = self.comm.project.config["mesh_file"]
        solver_settings = self.comm.project.solver_settings
        start_time = solver_settings["start_time"]
        end_time = solver_settings["end_time"]
        time_step = solver_settings["time_increment"]
        num_absorbing_layers = solver_settings["number_of_absorbing_layers"]
        polynomial_order = solver_settings["polynomial_order"]

        possible_boundaries = set(("r0", "t0", "t1", "p0", "p1",
                                   "inner_boundary"))
        absorbing_boundaries = \
            possible_boundaries.intersection(
                set(self.comm.project.domain.get_side_set_names()))
        if absorbing_boundaries:
            absorbing_boundaries = ",".join(sorted(absorbing_boundaries))
            print("Automatically determined the following absorbing "
                  "boundary side sets: %s" % absorbing_boundaries)

        salvus_command = \
            f"mpirun -n 4 --dimension 3 --mesh-file {mesh_file} " \
            f"--model-file {mesh_file} --start-time {start_time} " \
            f"--time-step {time_step} " \
            f"--end-time {end_time} --polynomial-order {polynomial_order} " \
            f"--adjoint --kernel-file kernel_{event_name}.e " \
            f"--load-fields adjoint " \
            f"--load-wavefield-file wavefield.h5 " \
            f"--io-memory-per-rank-in-MB 5000 " \
            f"--absorbing-boundaries {absorbing_boundaries} " \
            f"--source-toml {toml_file_name} " \
            f"--io-file-format bin"

        if self.comm.project.solver_settings["with_anisotropy"]:
            salvus_command += " --with-anisotropy --kernel-fields TTI"
        else:
            salvus_command += " --kernel-fields VP,VS,RHO"

        if num_absorbing_layers > 0:
            salvus_command += f" --num-absorbing-layers {num_absorbing_layers}"

        if self.comm.project.solver_settings["with_attenuation"]:
            salvus_command += f" --with-attenuation"

        salvus_command_file = os.path.join(output_dir, "run_salvus.sh")
        with open(salvus_command_file, "w") as fh:
            fh.write(salvus_command)

    # def make_event_mesh(self, event):
    #     """
    #     Make a specific mesh for an event. Uses location of event to
    #     structure the mesh in a specific way to optimise simulations.
    #     :param event: name of event
    #     """
    #     from lasif.tools.global_mesh_smoothiesem import mesh
    #
    #     n_axial_mask = 8
    #     n_lat = 12
    #     r_icb = 1221.5 / 6371.
    #
    #     event = self.comm.events.get(event)
    #
    #     src_lat = event["latitude"]
    #     src_lon = event["longitude"]
    #
    #     src_azimuth = 0.0
    #
    #     # Do I want to relate these to iterations?
    #     output_folder = os.path.join(self.comm.project.paths["models"],
    #                                  "EVENT_SPECIFIC", event["event_name"])
    #     if not os.path.exists(output_folder):
    #         os.makedirs(output_folder)
    #     output_filename = os.path.join(output_folder, "mesh.e")
    #     axisem_mesh_params = "file.yaml"  # This will be referred to later.
    #
    #     theta_min_lat_refine = []
    #     theta_max_lat_refine = []
    #     r_min_lat_refine = []
    #
    #     mesh(n_axial_mask=n_axial_mask, n_lat=n_lat, r_icb=r_icb,
    #          src_lat=src_lat,
    #          src_lon=src_lon, theta_min_lat_refine=theta_min_lat_refine,
    #          theta_max_lat_refine=theta_max_lat_refine,
    #          r_min_lat_refine=r_min_lat_refine,
    #          axisem_mesh_params=axisem_mesh_params,
    #          output_filename=output_filename,
    #          convert_element_type='tensorized', src_azimuth=src_azimuth)
    # def region_of_interest(self, event, mesh_file, radius):
    #     """
    #     Add to mesh file a region of interest field which is used when
    #     computing the gradient. The gradient will only be calculated
    #     inside the region of interest. This function currently removes the
    #     source imprint.
    #     :param event: The name of the event
    #     :param mesh_file: Where is the mesh which is used
    #     :param radius: Radius in kilometers for the region of interest
    #     :return: The mesh will include a binary ROI field.
    #     """
    #     from scipy.spatial import cKDTree
    #     from csemlib.io.exodus_reader import ExodusReader
    #     from csemlib.utils import sph2cart
    #
    #     event = self.comm.events.get(event)
    #     radius *= 1000.0  # Convert to meters.
    #
    #     src_colat = 90.0 - event["latitude"]
    #     src_lon = event["longitude"]
    #     src_depth_in_m = event["depth_in_km"] * 1000.0
    #     R_E = 6371000
    #
    #     b_x, b_y, b_z = sph2cart(np.radians(src_colat), np.radians(src_lon),
    #                              R_E - src_depth_in_m)
    #     src_point = [b_x, b_y, b_z]
    #
    #     e = ExodusReader(mesh_file)
    #     e_centroids = e.get_element_centroid()
    #     tree = cKDTree(e_centroids)
    #     idx = []
    #
    #     idx += tree.query_ball_point(src_point, radius)
    #
    #     e.close()
    #
    #     import pyexodus
    #
    #     e = pyexodus.exodus(mesh_file, "a")
    #     indices = list(idx)
    #
    #     e.put_element_variable_name("ROI", 1)
    #     e.put_element_variable_values(1, "ROI", 1, np.ones(len(e_centroids)))
    #     region_of_interest = e.get_element_variable_values(1, "ROI", 1)
    #
    #     region_of_interest[indices] = 0
    #     e.put_element_variable_values(1, "ROI", 1, region_of_interest)
    #     e.close()

    def get_sources_and_receivers_for_event(self, event_name: str,
                                            stf_filename: pathlib.Path):
        """
        Get a dictionary with all sources and receivers.

        :param iteration_name: The name of the iteration.
        """
        # In-method imports for to improve start-up speed.
        import pyasdf  # NOQA
        from salvus_mesh.unstructured_mesh import UnstructuredMesh
        import salvus_seismo.api  # NOQA

        asdf_filename = self.comm.waveforms.get_asdf_filename(
            event_name=event_name, data_type="raw")

        with pyasdf.ASDFDataSet(asdf_filename) as ds:
            event = ds.events[0]

            # Build inventory of all stations present in ASDF file
            stations = ds.waveforms.list()
            try:
                inv = ds.waveforms[stations[0]].StationXML
            except:
                continue
            for station in stations[1:]:
                inv += ds.waveforms[station].StationXML

        sources = [salvus_seismo.Source.parse(event)]
        receivers = salvus_seismo.Receiver.parse(inv)

        # Place sources and receivers exactly relative to the surface.
        src, rec = salvus_seismo.api.generate_sources_and_receivers(
            mesh=UnstructuredMesh.from_exodus(
                self.comm.project.config["mesh_file"]),
            sources=sources, receivers=receivers,
            # XXX: Set to False once moving to HDF5 models.
            is_exodus_mesh=True
            )

        # XXX: SINGLE SOURCE!!!
        s = self._write_custom_stf(
            source_dict=src["physics"]["wave-equation"]["point-source"][0],
            output_filename=stf_filename)

        src["physics"]["wave-equation"]["point-source"] = s

        return src, rec
