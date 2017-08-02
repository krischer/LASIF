#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import mock
import numpy as np
import os
import pytest
import shutil

from lasif import LASIFError
from lasif.components.project import Project
from lasif import rotations


@pytest.fixture()
def comm(tmpdir):
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "ExampleProject")
    tmpdir = str(tmpdir)
    shutil.copytree(proj_dir, os.path.join(tmpdir, "proj"))
    proj_dir = os.path.join(tmpdir, "proj")

    project = Project(project_root_path=proj_dir, init_project=False)

    return project.comm


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_input_files_are_actually_generated(patch, comm):
    """
    Tests if the input file generation actually creates some files and works in
    the first place.

    Does not test the input files. That is the responsibility of the input file
    generator module.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

    assert os.listdir(comm.project.paths["output"]) == []
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)

    output = os.path.join(comm.project.paths["output"], "input_files")

    # Normal simulation.
    comm.actions.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "normal simulation")
    output_dir = [_i for _i in os.listdir(output)
                  if "normal_simulation" in _i][0]
    assert len(os.listdir(os.path.join(output, output_dir))) != 0

    # Adjoint forward.
    comm.actions.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "adjoint forward")
    output_dir = [_i for _i in os.listdir(output)
                  if "adjoint_forward" in _i][0]
    assert len(os.listdir(os.path.join(output, output_dir))) != 0

    # Adjoint reverse.
    comm.actions.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "adjoint reverse")
    output_dir = [_i for _i in os.listdir(output)
                  if "adjoint_reverse" in _i][0]
    assert len(os.listdir(os.path.join(output, output_dir))) != 0


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_iteration_handling(patch, comm):
    """
    Tests the managing of the iterations.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

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
    assert iteration.data_preprocessing["lowpass_period"] == 8.0
    assert iteration.data_preprocessing["highpass_period"] == 100.0

    # Assert the processing parameters. This is somewhat redundant and should
    # rather be tested in the iteration test suite.
    process_params = iteration.get_process_params()
    assert process_params["npts"] == 500
    assert process_params["dt"] == 0.75
    assert process_params["lowpass"] == 0.125
    assert process_params["highpass"] == 0.01


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_generating_new_iteration(patch, comm):
    """
    Tests that iteration creation works.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

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


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_preprocessing_runs(patch, comm):
    """
    Simple tests to assure the preprocessing actually runs. Does not test if it
    does the right thing but will at least assure the program flow works as
    expected.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    processing_tag = comm.iterations.get("1").processing_tag
    event_data_dir = os.path.join(comm.project.paths["data"],
                                  "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    processing_dir = os.path.join(event_data_dir, processing_tag)
    assert not os.path.exists(processing_dir)
    # This will process only one event.
    comm.actions.preprocess_data(
        "1", ["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"])
    assert os.path.exists(processing_dir)
    assert len(os.listdir(processing_dir)) == 6

    # Remove and try again, this time not specifying the event which will
    # simply use all events. Should have the same result.
    shutil.rmtree(processing_dir)
    assert not os.path.exists(processing_dir)
    comm.actions.preprocess_data("1")
    assert os.path.exists(processing_dir)
    assert len(os.listdir(processing_dir)) == 6


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_finalize_adjoint_sources_with_failing_adjoint_src_calculation(
        patch, comm, capsys):
    """
    Tests the finalization of adjoint sources with a failing adjoint source
    calculation.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

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
    comm.actions.preprocess_data(it)

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
    assert "Could not create a single adjoint source." in out

    # Make sure nothing is actually written.
    out = os.path.join(comm.project.paths["output"], "adjoint_sources")
    adj_src_dir = os.path.join(out, os.listdir(out)[0])
    assert os.path.exists(adj_src_dir)
    assert len(os.listdir(adj_src_dir)) == 0


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_adjoint_source_finalization_unrotated_domain(patch, comm, capsys):
    """
    Tests the adjoint source finalization with an unrotated domain.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

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
    np.random.seed(123456)
    for station in stations:
        s = comm.waveforms.get_waveforms_synthetic(event_name, station,
                                                   it.long_name)
        # Perturb data a bit.
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

    # Make sure nothing is rotated as the domain is not rotated.
    with mock.patch("lasif.rotations.rotate_data") as patch:
        comm.actions.finalize_adjoint_sources(it.name, event_name)
    assert patch.call_count == 0

    out, _ = capsys.readouterr()
    assert "Wrote adjoint sources for 1 station(s)" in out

    # Make sure nothing is actually written.
    out = os.path.join(comm.project.paths["output"], "adjoint_sources")
    adj_src_dir = os.path.join(out, os.listdir(out)[0])
    assert os.path.exists(adj_src_dir)
    assert sorted(os.listdir(adj_src_dir)) == sorted(["ad_srcfile",
                                                      "ad_src_1"])


def test_adjoint_source_finalization_global_domain(comm, capsys):
    """
    Tests the adjoint source finalization with with a global domain.
    """
    from lasif.domain import GlobalDomain

    comm.project._component.domain = GlobalDomain()
    comm.iterations.create_new_iteration(
        "1", "specfem3d_globe_cem",
        comm.query.get_stations_for_all_events(), 8, 100)

    event_name = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    event = comm.events.get(event_name)
    t = event["origin_time"]

    # Create iteration.
    it = comm.iterations.get("1")

    # Fake preprocessed data by copying the synthetics and perturbing them a
    # bit...
    stations = ["HL.ARG", "HT.SIGR"]
    np.random.seed(123456)
    for station in stations:
        s = comm.waveforms.get_waveforms_synthetic(event_name, station,
                                                   it.long_name)
        # Perturb data a bit.
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

    # Make sure nothing is rotated as the domain is not rotated.
    with mock.patch("lasif.rotations.rotate_data") as patch:
        comm.actions.finalize_adjoint_sources(it.name, event_name)
    assert patch.call_count == 0

    out, _ = capsys.readouterr()
    assert "Wrote adjoint sources for 1 station(s)" in out

    # Make sure nothing is actually written.
    out = os.path.join(comm.project.paths["output"], "adjoint_sources")
    adj_src_dir = os.path.join(out, os.listdir(out)[0])
    assert os.path.exists(adj_src_dir)
    assert sorted(os.listdir(adj_src_dir)) == sorted(
        ["HL.ARG.MXE.adj", "HL.ARG.MXN.adj", "HL.ARG.MXZ.adj",
         "STATIONS_ADJOINT"])


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_adjoint_source_finalization_rotated_domain(patch, comm, capsys):
    """
    Tests the adjoint source finalization with a rotated domain.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

    # Set some rotation angle to actually get some rotated things.
    comm.project.domain.rotation_angle_in_degree = 0.1

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
    np.random.seed(123456)
    for station in stations:
        s = comm.waveforms.get_waveforms_synthetic(event_name, station,
                                                   it.long_name)
        # Perturb data a bit.
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

    # Make sure nothing is rotated as the domain is not rotated.
    rotate_data = rotations.rotate_data
    with mock.patch("lasif.rotations.rotate_data") as patch:
        patch.side_effect = \
            lambda *args, **kwargs: rotate_data(*args, **kwargs)
        comm.actions.finalize_adjoint_sources(it.name, event_name)
    # Once for each synthetic and once for the adjoint source.
    assert patch.call_count == 4

    out, _ = capsys.readouterr()
    assert "Wrote adjoint sources for 1 station(s)" in out

    # Make sure nothing is actually written.
    out = os.path.join(comm.project.paths["output"], "adjoint_sources")
    adj_src_dir = os.path.join(out, os.listdir(out)[0])

    assert os.path.exists(adj_src_dir)
    assert sorted(os.listdir(adj_src_dir)) == sorted(["ad_srcfile",
                                                      "ad_src_1"])


def test_adjoint_source_finalization_rotated_domain_specfem(comm, capsys):
    """
    Tests the adjoint source finalization with a rotated domain with SPECFEM.
    The difference here is that SPECFEM does not require rotations of the
    synthetics, and the adjoint sources..
    """
    # Set some rotation angle to actually get some rotated things.
    comm.project.domain.rotation_angle_in_degree = 0.1

    comm.iterations.create_new_iteration(
        "1", "specfem3d_globe_cem",
        comm.query.get_stations_for_all_events(), 8, 100)

    event_name = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    event = comm.events.get(event_name)
    t = event["origin_time"]

    # Create iteration.
    it = comm.iterations.get("1")

    # Fake preprocessed data by copying the synthetics and perturbing them a
    # bit...
    stations = ["HL.ARG", "HT.SIGR"]
    np.random.seed(123456)
    for station in stations:
        s = comm.waveforms.get_waveforms_synthetic(event_name, station,
                                                   it.long_name)
        # Perturb data a bit.
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

    # Make sure nothing is rotated as the domain is not rotated.
    rotate_data = rotations.rotate_data
    with mock.patch("lasif.rotations.rotate_data") as patch:
        patch.side_effect = \
            lambda *args, **kwargs: rotate_data(*args, **kwargs)
        comm.actions.finalize_adjoint_sources(it.name, event_name)
    # Should not be rotated at all!
    assert patch.call_count == 0

    out, _ = capsys.readouterr()
    assert "Wrote adjoint sources for 1 station(s) to" in out

    # Make sure nothing is actually written.
    out = os.path.join(comm.project.paths["output"], "adjoint_sources")
    adj_src_dir = os.path.join(out, os.listdir(out)[0])
    assert os.path.exists(adj_src_dir)
    assert sorted(os.listdir(adj_src_dir)) == sorted(
        ["HL.ARG.MXE.adj", "HL.ARG.MXN.adj", "HL.ARG.MXZ.adj",
         "STATIONS_ADJOINT"])


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_calculate_all_adjoint_sources_with_failing_adjoint_src_calculation(
        patch, comm, capsys):
    """
    Tests the calculates of adjoint sources with a failing adjoint source
    calculation.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

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
    comm.actions.preprocess_data(it)

    window_group_manager = comm.windows.get(event, it)

    # Automatic window selection does not work for the terrible test data...
    # Now add only windows that actually have data and synthetics but the
    # data will be too bad to actually extract an adjoint source from.
    for chan in ["HL.ARG..BHE", "HL.ARG..BHN", "HL.ARG..BHZ"]:
        window_group = window_group_manager.get(chan)
        window_group.add_window(starttime=t + 100, endtime=t + 200)
        window_group.write()

    capsys.readouterr()
    comm.actions.calculate_all_adjoint_sources(it.name, event_name)
    out, _ = capsys.readouterr()
    assert "Could not calculate adjoint source for iteration 1" in out

    # Make sure nothing is actually written.
    out = os.path.join(comm.project.paths["adjoint_sources"], event_name,
                       it.long_name)
    assert os.listdir(out) == []


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_calculate_all_adjoint_sources_rotated_domain(patch, comm, capsys):
    """
    Tests the adjoint source calculation with a rotated domain.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

    # Set some rotation angle to actually get some rotated things.
    comm.project.domain.rotation_angle_in_degree = 0.1

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
    np.random.seed(123456)
    for station in stations:
        s = comm.waveforms.get_waveforms_synthetic(event_name, station,
                                                   it.long_name)
        # Perturb data a bit.
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

    # Make sure nothing is rotated as the domain is not rotated.
    rotate_data = rotations.rotate_data
    with mock.patch("lasif.rotations.rotate_data") as patch:
        patch.side_effect = \
            lambda *args, **kwargs: rotate_data(*args, **kwargs)
        comm.actions.calculate_all_adjoint_sources(it.name, event_name)
    # Once for each synthetic.
    assert patch.call_count == 3

    out, _ = capsys.readouterr()
    assert out == ""

    # Make sure that three adjoint sources are written in the end.
    out = os.path.join(comm.project.paths["adjoint_sources"], event_name,
                       it.long_name)
    # Joblib dumps one or two files per written array, depending on the
    # version.
    assert len(os.listdir(out)) >= 3
