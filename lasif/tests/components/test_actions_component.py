#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import numpy as np
import os
import pytest
import shutil

from lasif import LASIFError, LASIFNotFoundError
from lasif.components.project import Project


@pytest.fixture()
def comm(tmpdir):
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "ExampleProject")
    tmpdir = str(tmpdir)
    shutil.copytree(proj_dir, os.path.join(tmpdir, "proj"))
    proj_dir = os.path.join(tmpdir, "proj")

    project = Project(project_root_path=proj_dir, init_project=False)

    return project.comm


def test_input_files_are_actually_generated(comm):
    """
    Tests if the input file generation actually creates some files and works in
    the first place.

    Does not test the input files. That is the responsibility of the input file
    generator module.
    """
    assert os.listdir(comm.project.paths["output"]) == []
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)

    # Normal simulation.
    comm.actions.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "normal simulation")
    output_dir = [_i for _i in os.listdir(comm.project.paths["output"])
                  if "normal_simulation" in _i][0]
    assert len(os.listdir(os.path.join(
        comm.project.paths["output"], output_dir))) != 0

    # Adjoint forward.
    comm.actions.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "adjoint forward")
    output_dir = [_i for _i in os.listdir(comm.project.paths["output"])
                  if "adjoint_forward" in _i][0]
    assert len(os.listdir(os.path.join(
        comm.project.paths["output"], output_dir))) != 0

    # Adjoint reverse.
    comm.actions.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "adjoint reverse")
    output_dir = [_i for _i in os.listdir(comm.project.paths["output"])
                  if "adjoint_reverse" in _i][0]
    assert len(os.listdir(os.path.join(
        comm.project.paths["output"], output_dir))) != 0


def test_iteration_handling(comm):
    """
    Tests the managing of the iterations.
    """
    # First create two iterations.
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    comm.iterations.create_new_iteration(
        "2", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    assert sorted(os.listdir(comm.project.paths["iterations"])) == \
        sorted(["ITERATION_1.xml", "ITERATION_2.xml"])

    assert sorted(comm.iterations.list()) == sorted(["1", "2"])

    iteration = comm.iterations.get("1")
    # Assert that the aspects of the example project did get picked up by the
    # iteration. Only one event will be available as the other is empty.
    assert len(iteration.events) == 1

    assert len(iteration.events["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"][
               "stations"]) == 4
    assert iteration.iteration_name == "1"
    assert iteration.source_time_function == "Filtered Heaviside"
    assert iteration.data_preprocessing["lowpass_period"] == 8.0
    assert iteration.data_preprocessing["highpass_period"] == 100.0

    # Assert the processing parameters. This is somewhat redundant and should
    # rather be tested in the iteration test suite.
    process_params = iteration.get_process_params()
    assert process_params["npts"] == 500
    assert process_params["dt"] == 0.75
    assert process_params["stf"] == "Filtered Heaviside"
    assert process_params["lowpass"] == 0.125
    assert process_params["highpass"] == 0.01


def test_generating_new_iteration(comm):
    """
    Tests that iteration creation works.
    """
    assert os.listdir(comm.project.paths["iterations"]) == []

    # Using an invalid solver raises.
    with pytest.raises(LASIFError) as excinfo:
        comm.iterations.create_new_iteration(
            "1", "unknown", comm.query.get_stations_for_all_events(), 8, 100)
    msg = excinfo.value.message
    assert "not known" in msg
    assert "unknown" in msg

    # Nothing should have happened.
    assert os.listdir(comm.project.paths["iterations"]) == []

    # Now actually create a new iteration.
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    assert os.listdir(comm.project.paths["iterations"]) == ["ITERATION_1.xml"]

    # Creating an already existing iteration raises.
    with pytest.raises(LASIFError) as excinfo:
        comm.iterations.create_new_iteration(
            "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    assert excinfo.value.message.lower() == "iteration 1 already exists."


def test_preprocessing_runs(comm):
    """
    Simple tests to assure the preprocessing actually runs. Does not test if it
    does the right thing but will at least assure the program flow works as
    expected.
    """
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    processing_tag = comm.iterations.get("1").processing_tag
    event_data_dir = os.path.join(comm.project.paths["data"],
                                  "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    processing_dir = os.path.join(event_data_dir, processing_tag)
    assert not os.path.exists(processing_dir)
    # This will process only one event.
    comm.actions.preprocess_data(
        "1", ["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"], waiting_time=0.0)
    assert os.path.exists(processing_dir)
    assert len(os.listdir(processing_dir)) == 6

    # Remove and try again, this time not specifying the event which will
    # simply use all events. Should have the same result.
    shutil.rmtree(processing_dir)
    assert not os.path.exists(processing_dir)
    comm.actions.preprocess_data("1", waiting_time=0.0)
    assert os.path.exists(processing_dir)
    assert len(os.listdir(processing_dir)) == 6


def test_finalize_adjoint_sources_with_failing_adjoint_src_calculation(
        comm, capsys):
    """
    Tests the finalization of adjoint sources with a failing adjoint source
    calculation.
    """
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)

    event_name = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    event = comm.events.get(event_name)
    t = event["origin_time"]

    # Create iteration and preprocess some data.
    it = comm.iterations.get("1")
    # Make sure the settings match the synthetics.
    it.solver_settings["solver_settings"]["simulation_parameters"][
        "time_increment"] = 0.13
    it.solver_settings["solver_settings"]["simulation_parameters"][
        "number_of_time_steps"] = 4000
    comm.actions.preprocess_data(it, waiting_time=0.0)

    window_group_manager = comm.windows.get(event, it)

    # Automatic window selection does not work for the terrible test data...
    # Now add only windows that actually have data and synthetics but the
    # data will be too bad to actually extract an adjoint source from.
    for chan in ["HL.ARG..BHE", "HL.ARG..BHN", "HL.ARG..BHZ"]:
        window_group = window_group_manager.get(chan)
        window_group.add_window(starttime=t + 100, endtime=t + 200)
        window_group.write()

    capsys.readouterr()
    comm.actions.finalize_adjoint_sources(it.name, event_name)
    out, _ = capsys.readouterr()
    assert "Could not calculate adjoint source for iteration 1" in out
    assert "Wrote 0 adjoint sources" in out

    # Make sure nothing is actually written.
    out = comm.project.paths["output"]
    adj_src_dir = [i for i in os.listdir(out) if "adjoint_sources" in i][0]
    adj_src_dir = os.path.join(out, adj_src_dir)
    assert os.path.exists(adj_src_dir)
    assert len(os.listdir(adj_src_dir)) == 0


def test_adjoint_source_finalization_unrotated_domain(comm, capsys):
    """
    Tests the adjoint source finalization with an unrotated domain.
    """
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)

    event_name = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    event = comm.events.get(event_name)
    t = event["origin_time"]

    # Create iteration.
    it = comm.iterations.get("1")


    # Fake preprocessed data by copying the synthetics and perturbing them a
    # bit...
    stations = ["HL.ARG", "HT.SIGR"]
    for station in stations:
        s = comm.waveforms.get_waveforms_synthetic(event_name, station,
                                                   it.long_name)
        # Perturb data.
        for tr in s:
            tr.data += np.random.random(len(tr.data)) * 2E-8
        path = comm.waveforms.get_waveform_folder(event_name, "processed",
                                                  it.processing_tag)
        if not os.path.exists(path):
            os.makedirs(path)
        for tr in s:
            tr.write(os.path.join(path, tr.id), format="mseed")

    window_group_manager = comm.windows.get(event, it)

    # Automatic window selection does not work for the terrible test data...
    # Now add only windows that actually have data and synthetics but the
    # data will be too bad to actually extract an adjoint source from.
    for chan in ["HL.ARG..BHE", "HL.ARG..BHN", "HL.ARG..BHZ"]:
        window_group = window_group_manager.get(chan)
        window_group.add_window(starttime=t + 100, endtime=t + 200)
        window_group.write()

    capsys.readouterr()
    comm.actions.finalize_adjoint_sources(it.name, event_name)
    out, _ = capsys.readouterr()
    assert "Wrote 1 adjoint sources" in out

    # Make sure nothing is actually written.
    out = comm.project.paths["output"]
    adj_src_dir = [i for i in os.listdir(out) if "adjoint_sources" in i][0]
    adj_src_dir = os.path.join(out, adj_src_dir)
    assert os.path.exists(adj_src_dir)
    assert sorted(os.listdir(adj_src_dir)) == sorted(["ad_srcfile",
                                                      "ad_src_1"])
