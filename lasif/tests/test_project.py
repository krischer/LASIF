#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test cases for the project class.

Some of these tests are not actually verifying the results but rather assure
that all components of LASIF are able to work together so these test can be
viewed as integration tests.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import matplotlib as mpl
from lasif import LASIFError

mpl.use("agg")

import copy
import glob
import mock
import obspy
import os
import pytest
import shutil

from lasif import LASIFError
from lasif.project import Project
from lasif.tests.testing_helpers import project  # NOQA
from lasif.tests.testing_helpers import images_are_identical, \
    reset_matplotlib, DATA




def test_event_info_retrieval(project):
    """
    Test the dictionary retrieved from each event.

    The dictionary is used to extract information about a single event.
    """
    event_info = project.events["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"]
    assert event_info["latitude"] == 38.82
    assert event_info["longitude"] == 40.14
    assert event_info["depth_in_km"] == 4.5
    assert event_info["origin_time"] == \
        obspy.UTCDateTime(2010, 3, 24, 14, 11, 31)
    assert event_info["region"] == "TURKEY"
    assert event_info["magnitude"] == 5.1
    assert event_info["magnitude_type"] == "Mwc"

    event_info = project.events["GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"]
    assert event_info["latitude"] == 39.15
    assert event_info["longitude"] == 29.1
    assert event_info["depth_in_km"] == 7.0
    assert event_info["origin_time"] == \
        obspy.UTCDateTime(2011, 5, 19, 20, 15, 22, 900000)
    assert event_info["region"] == "TURKEY"
    assert event_info["magnitude"] == 5.9
    assert event_info["magnitude_type"] == "Mwc"


def test_event_info_filecount_retrieval(project):
    """
    Checks the filecount retrieval with the event_info method.
    """
    event_info = project.get_filecounts_for_event(
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    assert event_info["raw_waveform_file_count"] == 4
    assert event_info["preprocessed_waveform_file_count"] == 0
    assert event_info["synthetic_waveform_file_count"] == 6

    event_info = project.get_filecounts_for_event(
        "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15")
    assert event_info["raw_waveform_file_count"] == 0
    assert event_info["preprocessed_waveform_file_count"] == 0
    assert event_info["synthetic_waveform_file_count"] == 0


def test_waveform_cache_usage(project):
    """
    Tests the automatic creation and usage of the waveform caches.
    """
    event_name = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    waveform_cache = os.path.join(project.paths["data"], event_name,
                                  "raw_cache.sqlite")
    waveform_folder = os.path.join(project.paths["data"], event_name, "raw")

    # The example project does not yet have the cache.
    assert not os.path.exists(waveform_cache)

    # Create the cache.
    cache = project._get_waveform_cache_file(event_name, "raw",
                                             show_progress=False)

    # Make sure it now exists.
    assert os.path.exists(waveform_cache)

    # Accessing it again should be much faster as it already exists.
    cache = project._get_waveform_cache_file(event_name, "raw",
                                             show_progress=False)

    # The cache has to point to the correct folder and file.
    assert cache.waveform_folder == waveform_folder
    assert cache.cache_db_file == waveform_cache
    # Make sure the cache contains all files.
    assert sorted(cache.files["waveform"]) == \
        sorted([os.path.join(waveform_folder, _i)
                for _i in os.listdir(waveform_folder)])

    # Tests an exemplary file.
    filename = os.path.join(project.paths["data"], event_name, "raw",
                            "HL.ARG..BHZ.mseed")
    assert os.path.exists(filename)
    info = cache.get_details(filename)[0]
    assert info["network"] == "HL"
    assert info["station"] == "ARG"
    assert info["location"] == ""
    assert info["channel"] == "BHZ"
    # The file does not contain information about the location of the station.
    assert info["latitude"] is None
    assert info["longitude"] is None
    assert info["elevation_in_m"] is None
    assert info["local_depth_in_m"] is None


def test_station_filename_generator(project):
    """
    Make sure existing stations are not overwritten by creating unique new
    station filenames. This is used when downloading new station files.
    """
    new_seed_filename = \
        project.get_channel_filename("HL", "ARG", "", "BHZ", "datalessSEED")
    existing_seed_filename = glob.glob(os.path.join(
        project.paths["dataless_seed"], "dataless.HL_*"))[0]

    assert os.path.exists(existing_seed_filename)
    assert existing_seed_filename != new_seed_filename
    assert os.path.dirname(existing_seed_filename) == \
        os.path.dirname(new_seed_filename)
    assert os.path.dirname(new_seed_filename) == project.paths["dataless_seed"]

    # Test RESP file name generation.
    resp_filename_1 = project.get_channel_filename("A", "B", "C", "D", "RESP")
    assert not os.path.exists(resp_filename_1)
    assert os.path.dirname(resp_filename_1) == project.paths["resp"]
    with open(resp_filename_1, "wt") as fh:
        fh.write("blub")
    assert os.path.exists(resp_filename_1)

    resp_filename_2 = project.get_channel_filename("A", "B", "C", "D", "RESP")
    assert resp_filename_1 != resp_filename_2
    assert os.path.dirname(resp_filename_2) == project.paths["resp"]


def test_generating_new_iteration(project):
    """
    Tests that iteration creation works.
    """
    assert os.listdir(project.paths["iterations"]) == []

    # Using an invalid solver raises.
    with pytest.raises(LASIFError) as excinfo:
        project.create_new_iteration("1", "unknown_solver", 8, 100)
    msg = excinfo.value.message
    assert "not known" in msg
    assert "unknown_solver" in msg

    # Nothing should have happened.
    assert os.listdir(project.paths["iterations"]) == []

    # Now actually create a new iteration.
    project.create_new_iteration("1", "ses3d_4_1", 8, 100)
    assert os.listdir(project.paths["iterations"]) == ["ITERATION_1.xml"]

    # Creating an already existing iteration raises.
    with pytest.raises(LASIFError) as excinfo:
        project.create_new_iteration("1", "ses3d_4_1", 8, 100)
    assert excinfo.value.message.lower() == "iteration already exists."


def test_iteration_handling(project):
    """
    Tests the managing of the iterations.
    """
    # First create two iterations.
    project.create_new_iteration("1", "ses3d_4_1", 8, 100)
    project.create_new_iteration("2", "ses3d_4_1", 8, 100)
    assert sorted(os.listdir(project.paths["iterations"])) == \
        sorted(["ITERATION_1.xml", "ITERATION_2.xml"])

    # Make sure they are found correctly.
    assert project.get_iteration_dict() == {key: os.path.join(
        project.paths["iterations"], "ITERATION_" + key + ".xml")
        for key in ["1", "2"]}

    iteration = project._get_iteration("1")

    # Assert that the aspects of the example project did get picked up by the
    # iteration. Only one event will be available as the other is empty.
    assert len(iteration.events) == 1

    assert len(iteration.events["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"]
               ["stations"]) == 4
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


def test_preprocessing_runs(project):
    """
    Simple tests to assure the preprocessing actually runs. Does not test if it
    does the right thing but will at least assure the program flow works as
    expected.
    """
    project.create_new_iteration("1", "ses3d_4_1", 8, 100)
    processing_tag = project._get_iteration("1").processing_tag
    event_data_dir = os.path.join(
        project.paths["data"], "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    processing_dir = os.path.join(event_data_dir, processing_tag)
    assert not os.path.exists(processing_dir)
    # This will process only one event.
    project.preprocess_data("1", ["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"],
                            waiting_time=0.0)
    assert os.path.exists(processing_dir)
    assert len(os.listdir(processing_dir)) == 4

    # Remove and try again, this time not specifying the event which will
    # simply use all events. Should have the same result.
    shutil.rmtree(processing_dir)
    assert not os.path.exists(processing_dir)
    project.preprocess_data("1", waiting_time=0.0)
    assert os.path.exists(processing_dir)
    assert len(os.listdir(processing_dir)) == 4



def test_has_station_file(project):
    """
    Tests if the has_station_file_method().
    """
    assert project.has_station_file(
        "HL.ARG..BHZ", obspy.UTCDateTime(2010, 3, 24, 14, 30)) is True
    assert project.has_station_file(
        "HT.SIGR..HHZ", obspy.UTCDateTime(2010, 3, 24, 14, 30)) is True
    assert project.has_station_file(
        "KO.KULA..BHZ", obspy.UTCDateTime(2010, 3, 24, 14, 30)) is True
    assert project.has_station_file(
        "KO.RSDY..BHZ", obspy.UTCDateTime(2010, 3, 24, 14, 30)) is True

    assert project.has_station_file(
        "HL.ARG..BHZ", obspy.UTCDateTime(1970, 3, 24, 14, 30)) is False
    assert project.has_station_file(
        "HT.SIGR..HHZ", obspy.UTCDateTime(1970, 3, 24, 14, 30)) is False
    assert project.has_station_file(
        "KO.KULA..BHZ", obspy.UTCDateTime(1970, 3, 24, 14, 30)) is False
    assert project.has_station_file(
        "KO.RSDY..BHZ", obspy.UTCDateTime(1970, 3, 24, 14, 30)) is False


def test_input_file_invocation(project):
    """
    Tests if the input file generation actually creates some files and works in
    the first place.

    Does not test the input files. That is the responsibility of the input file
    generator module.
    """
    assert os.listdir(project.paths["output"]) == []
    project.create_new_iteration("1", "ses3d_4_1", 8, 100)

    # Normal simulation.
    project.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "normal simulation")
    output_dir = [_i for _i in os.listdir(project.paths["output"])
                  if "normal_simulation" in _i][0]
    assert len(os.listdir(os.path.join(
        project.paths["output"], output_dir))) != 0

    # Adjoint forward.
    project.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "adjoint forward")
    output_dir = [_i for _i in os.listdir(project.paths["output"])
                  if "adjoint_forward" in _i][0]
    assert len(os.listdir(os.path.join(
        project.paths["output"], output_dir))) != 0

    # Adjoint reverse.
    project.generate_input_files(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "adjoint reverse")
    output_dir = [_i for _i in os.listdir(project.paths["output"])
                  if "adjoint_reverse" in _i][0]
    assert len(os.listdir(os.path.join(
        project.paths["output"], output_dir))) != 0


def test_data_validation(project, capsys):
    """
    Attempt to test the data validation part in a simple manner.
    """
    # def reset():
    #     try:
    #         project.events = EventPseudoDict(project.paths["events"])
    #     except:
    #         pass
    #     try:
    #         obspy.core.event.ResourceIdentifier\
    #             ._ResourceIdentifier__resource_id_weak_dict.clear()
    #     except:
    #         pass
    def reset():
        pass

    # The default output should be fully valid.
    project.validate_data()
    out = capsys.readouterr()[0]
    assert "ALL CHECKS PASSED" in out
    reset()

    filename = os.path.join(project.paths["events"],
                            "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11.xml")
    with open(filename, "rt") as fh:
        original_contents = fh.read()

    reset()

    # Now make a faulty QuakeML file. Removing a public id will trigger an
    # error int he QuakeML validation.
    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            'publicID="smi:local/76499e98-9ac4-4de1-844b-4042d0e80915"', ""))
    project.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Validating against ")][0]
    assert "FAIL" in line
    reset()

    # Now duplicate an id.
    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            "smi:service.iris.edu/fdsnws/event/1/query?eventid=2847365",
            "smi:www.iris.edu/spudservice/momenttensor/gcmtid/"
            "C201003241411A#reforigin"))
    project.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Checking for duplicate ")][0]
    assert "FAIL" in line
    reset()

    # Now make the file have an insane depth. This should trigger a sanity
    # check error.
    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            "<value>4500.0</value>",
            "<value>450000000000.0</value>"))
    project.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Performing some basic sanity ")][0]
    assert "FAIL" in line
    reset()

    # Trigger an error that two events are too close in time.
    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            "2010-03-24T14:11:31.000000Z",
            "2011-05-19T20:15:22.900000Z"))
    project.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Checking for duplicates and ")][0]
    assert "FAIL" in line
    reset()

    # Create an event outside of the chosen domain.

    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            "<value>40.14</value>",
            "<value>-100.0</value>"))
    project.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Assure all events are in chosen")][0]
    assert "FAIL" in line
    reset()


def test_data_synthetic_iterator(project, recwarn):
    """
    Tests that the data synthetic iterator works as expected.
    """
    # It requires an existing iteration with processed data.
    project.create_new_iteration("1", "ses3d_4_1", 8, 100)
    project.preprocess_data("1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
                            waiting_time=0.0)

    iterator = project.data_synthetic_iterator(
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "1")

    # The example project only contains synthetics for two stations.
    expected = {
        "HL.ARG..BHZ": {"latitude": 36.216, "local_depth_in_m": 0.0,
                        "elevation_in_m": 170.0, "longitude": 28.126},
        "HT.SIGR..HHZ": {"latitude": 39.2114, "local_depth_in_m": 0.0,
                         "elevation_in_m": 93.0, "longitude": 25.8553}}

    found = []
    no_synthetics_found_count = 0

    assert recwarn.list == []

    for _i in iterator:
        if _i is None:
            # If the data is not found it should always warn.
            w = recwarn.pop(UserWarning)
            assert "No synthetics found" in w.message.args[0]
            no_synthetics_found_count += 1
            continue
        data = _i.data
        synth = _i.synthetics
        coods = _i.coordinates
        assert recwarn.list == []

        # Only one real data is present at all times.
        assert len(data) == 1
        # Three synthetic components.
        assert len(synth) == 3

        found.append(data[0].id)
        assert data[0].id in expected

        assert expected[data[0].id] == coods

    assert sorted(found) == sorted(expected.keys())
    # No synthetics exists for the other two.
    assert no_synthetics_found_count == 2


def test_string_representation(project, capsys):
    """
    Tests the projects string representation.
    """
    print(project)
    out = capsys.readouterr()[0]
    assert "\"ExampleProject\"" in out
    assert "Toy Project used in the Test Suite" in out
    assert "2 events" in out
    assert "4 station files" in out
    assert "4 raw waveform files" in out
    assert "0 processed waveform files" in out
    assert "6 synthetic waveform files" in out


def test_is_event_station_raypath_within_boundaries(project):
    """
    Tests the raypath checker.
    """
    # latitude = 38.82
    # longitude = 40.14
    event = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"

    assert project.is_event_station_raypath_within_boundaries(
        event, 38.92, 40.0)
    assert not project.is_event_station_raypath_within_boundaries(
        event, 38.92, 140.0)


def test_synthetic_waveform_finding(project):
    """
    Tests the synthetic filename finder.
    """
    ev = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    it = "1"
    stations = project._get_synthetic_waveform_filenames(ev, it)
    assert "HL.ARG" in stations
    assert "HT.SIGR" in stations
    assert "X" in stations["HL.ARG"]
    assert "Y" in stations["HL.ARG"]
    assert "Z" in stations["HL.ARG"]
    assert "X" in stations["HT.SIGR"]
    assert "Y" in stations["HT.SIGR"]
    assert "Z" in stations["HT.SIGR"]
    assert stations["HT.SIGR"]["Z"] == os.path.join(
        project.paths["synthetics"], ev, "ITERATION_" + it, "HT.SIGR_.___.z")

    stations = project._get_synthetic_waveform_filenames(ev, it + "2")
    assert stations == {}


def test_iteration_status(project):
    """
    Tests the iteration status commands.
    """
    project.create_new_iteration("1", "ses3d_4_1", 8, 100)
    event = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"

    # Currenty the project has 4 files, that are not preprocessed.
    status = project.get_iteration_status("1")
    assert len(status["channels_not_yet_preprocessed"]) == 4
    assert status["stations_in_iteration_that_do_not_exist"] == []
    # The project only has synthetics for two stations.
    assert sorted(status["synthetic_data_missing"][event]) == ["KO.KULA",
                                                               "KO.RSDY"]

    # Preprocess some files.
    project.preprocess_data("1", [event],
                            waiting_time=0.0)

    status = project.get_iteration_status("1")
    assert status["channels_not_yet_preprocessed"] == []
    assert status["stations_in_iteration_that_do_not_exist"] == []
    assert sorted(status["synthetic_data_missing"][event]) == ["KO.KULA",
                                                               "KO.RSDY"]

    # Remove one of the waveform files. This has the effect that the iteration
    # contains a file that is not actually in existance. This should be
    # detected.
    proc_folder = os.path.join(
        project.paths["data"], event,
        project._get_iteration("1").processing_tag)
    data_folder = os.path.join(project.paths["data"], event, "raw")

    data_file = sorted(glob.glob(os.path.join(data_folder, "*")))[0]
    proc_file = sorted(glob.glob(os.path.join(proc_folder, "*")))[0]
    os.remove(data_file)
    os.remove(proc_file)

    status = project.get_iteration_status("1")
    assert status["channels_not_yet_preprocessed"] == []
    assert len(status["stations_in_iteration_that_do_not_exist"]) == 1
    assert sorted(status["synthetic_data_missing"][event]) == ["KO.KULA",
                                                               "KO.RSDY"]

    # Now remove all synthetics. This should have the result that all
    # synthetics are missing.
    for folder in os.listdir(project.paths["synthetics"]):
        shutil.rmtree(os.path.join(project.paths["synthetics"], folder))
    status = project.get_iteration_status("1")
    assert status["channels_not_yet_preprocessed"] == []
    assert len(status["stations_in_iteration_that_do_not_exist"]) == 1
    # HL.ARG has been remove before.
    assert sorted(status["synthetic_data_missing"][event]) == \
        ["HT.SIGR", "KO.KULA", "KO.RSDY"]


