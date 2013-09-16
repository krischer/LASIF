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
mpl.use("agg")

import copy
import glob
import inspect
import matplotlib.pylab as plt
from matplotlib.testing.compare import compare_images as mpl_compare_images
import obspy
import os
import pytest
import shutil

from lasif.project import Project, LASIFException


# Folder where all the images for comparison are stored.
IMAGES = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "baseline_images")
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


@pytest.fixture
def project(tmpdir):
    """
    Fixture returning the initialized example project. It will be a fresh copy
    every time that will be deleted after the test has finished so every test
    can mess with the contents of the folder.
    """
    # A new project will be created many times. ObsPy complains if objects that
    # already exists are created again.
    obspy.core.event.ResourceIdentifier\
        ._ResourceIdentifier__resource_id_weak_dict.clear()

    # Copy the example project
    example_project = os.path.join(DATA, "ExampleProject")
    project_path = os.path.join(str(tmpdir), "ExampleProject")
    shutil.copytree(example_project, project_path)

    # Init it. This will create the missing paths.
    return Project(project_path)


def setup_function(function):
    """
    Make sure matplotlib behaves the same on every machine.
    """
    # Set all default values.
    mpl.rcdefaults()
    # These settings must be hardcoded for running the comparision tests and
    # are not necessarily the default values.
    mpl.rcParams['font.family'] = 'Bitstream Vera Sans'
    mpl.rcParams['text.hinting'] = False
    # Not available for all matplotlib versions.
    try:
        mpl.rcParams['text.hinting_factor'] = 8
    except KeyError:
        pass
    import locale
    locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))


def images_are_identical(image_name, temp_dir, dpi=None):
    """
    Partially copied from ObsPy
    """
    image_name += os.path.extsep + "png"
    expected = os.path.join(IMAGES, image_name)
    actual = os.path.join(temp_dir, image_name)

    if dpi:
        plt.savefig(actual, dpi=dpi)
    else:
        plt.savefig(actual)
    plt.close()

    assert os.path.exists(expected)
    assert os.path.exists(actual)

    # Use a reasonably high tolerance to get around difference with different
    # freetype and possibly agg versions. matplotlib uses a tolerance of 13.
    result = mpl_compare_images(expected, actual, 5, in_decorator=True)
    if result is not None:
        print result
    assert result is None


def test_initalizing_project_in_wrong_path():
    """
    A useful error message should be returned when a project initialized with
    the wrong path.
    """
    with pytest.raises(LASIFException) as excinfo:
        Project("/some/random/path")
    assert "wrong project path" in excinfo.value.message.lower()


def test_project_creation(tmpdir):
    """
    Tests the project creation.
    """
    tmpdir = str(tmpdir)

    pr = Project(tmpdir, init_project="TestProject")

    # Make sure the root path is correct.
    assert pr.paths["root"] == tmpdir

    # Check that all folders did get created.
    foldernames = [
        "EVENTS",
        "DATA",
        "CACHE",
        "LOGS",
        "MODELS",
        "ITERATIONS",
        "SYNTHETICS",
        "KERNELS",
        "STATIONS",
        "OUTPUT",
        os.path.join("ADJOINT_SOURCES_AND_WINDOWS", "WINDOWS"),
        os.path.join("ADJOINT_SOURCES_AND_WINDOWS", "ADJOINT_SOURCES"),
        os.path.join("STATIONS", "SEED"),
        os.path.join("STATIONS", "StationXML"),
        os.path.join("STATIONS", "RESP")]
    for foldername in foldernames:
        assert os.path.isdir(os.path.join(tmpdir, foldername))

    # Make some more tests that the paths are set correctly.
    assert pr.paths["events"] == os.path.join(tmpdir, "EVENTS")
    assert pr.paths["kernels"] == os.path.join(tmpdir, "KERNELS")

    # Assert that the config file has been created.
    assert os.path.exists(os.path.join(pr.paths["root"], "config.xml"))


def test_config_file_creation_and_parsing(tmpdir):
    """
    Tests the creation of a default config file and the reading of the file.
    """
    # Create a new project.
    pr = Project(str(tmpdir), init_project="TestProject")
    del pr

    # Init it once again.
    pr = Project(str(tmpdir))

    # Assert the config file will test the creation of the default file and the
    # reading.
    assert pr.config["name"] == "TestProject"
    assert pr.config["description"] == ""
    assert pr.config["download_settings"]["arclink_username"] is None
    assert pr.config["download_settings"]["seconds_before_event"] == 300
    assert pr.config["download_settings"]["seconds_after_event"] == 3600
    assert pr.domain["bounds"]["minimum_longitude"] == -20
    assert pr.domain["bounds"]["maximum_longitude"] == 20
    assert pr.domain["bounds"]["minimum_latitude"] == -20
    assert pr.domain["bounds"]["maximum_latitude"] == 20
    assert pr.domain["bounds"]["minimum_depth_in_km"] == 0.0
    assert pr.domain["bounds"]["maximum_depth_in_km"] == 200.0
    assert pr.domain["bounds"]["boundary_width_in_degree"] == 3.0
    assert pr.domain["rotation_axis"] == [1.0, 1.0, 1.0]
    assert pr.domain["rotation_angle"] == -45.0


def test_config_file_caching(tmpdir):
    """
    The config file is cached to read if faster as it is read is every single
    time a LASIF command is executed.
    """
    # Create a new project.
    pr = Project(str(tmpdir), init_project="TestProject")
    config = copy.deepcopy(pr.config)
    domain = copy.deepcopy(pr.domain)
    del pr

    # Check that the config file cache has been created.
    cache = os.path.join(str(tmpdir), "CACHE", "config.xml_cache.pickle")
    assert os.path.exists(cache)

    # Delete it.
    os.remove(cache)
    assert not os.path.exists(cache)

    pr = Project(str(tmpdir), init_project="TestProject")

    # Assert that everything is still the same.
    assert config == pr.config
    assert domain == pr.domain
    del pr

    # This should have created the cached file.
    assert os.path.exists(cache)

    pr = Project(str(tmpdir), init_project="TestProject")

    # Assert that nothing changed.
    assert config == pr.config
    assert domain == pr.domain


def test_domain_plotting(tmpdir):
    """
    Very simple domain plotting test.
    """
    pr = Project(str(tmpdir), init_project="TestProject")
    pr.plot_domain(show_plot=False)

    images_are_identical("simple_test_domain", str(tmpdir))


def test_event_handling(project):
    """
    Tests the event handling.
    """
    # Get the event dictionary. This is a simple mapping between internal event
    # name and QuakeML filename.
    event_dict = project.get_event_dict()
    assert len(event_dict) == 2
    assert "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15" in event_dict
    assert event_dict["GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"].endswith(
        "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15.xml")
    assert event_dict["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"].endswith(
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11.xml")

    # Test single and multiple event retrieval.
    event_1 = project.get_event("GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15")
    event_2 = project.get_event("GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    assert isinstance(event_1, obspy.core.event.Event)
    assert isinstance(event_2, obspy.core.event.Event)
    events = sorted([event_1, event_2])
    assert events == sorted(project.get_all_events())


def test_event_plotting(project):
    """
    Tests the plotting of all events.

    The commands supports three types of plots: Beachballs on a map and depth
    and time distribution histograms.
    """
    project.plot_events(plot_type="map", show_plot=False)
    images_are_identical("two_events_plot_map", project.paths["root"])

    project.plot_events(plot_type="depth", show_plot=False)
    images_are_identical("two_events_plot_depth", project.paths["root"])

    project.plot_events(plot_type="time", show_plot=False)
    images_are_identical("two_events_plot_time", project.paths["root"])


def test_event_info_retrieval(project):
    """
    Test the dictionary retrieved from each event.

    The dictionary is used to extract information about a single event.
    """
    event_info = project.get_event_info(
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    assert event_info["latitude"] == 38.82
    assert event_info["longitude"] == 40.14
    assert event_info["depth_in_km"] == 4.5
    assert event_info["origin_time"] == \
        obspy.UTCDateTime(2010, 3, 24, 14, 11, 31)
    assert event_info["region"] == "TURKEY"
    assert event_info["magnitude"] == 5.1
    assert event_info["magnitude_type"] == "Mwc"

    event_info = project.get_event_info(
        "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15")
    assert event_info["latitude"] == 39.15
    assert event_info["longitude"] == 29.1
    assert event_info["depth_in_km"] == 7.0
    assert event_info["origin_time"] == \
        obspy.UTCDateTime(2011, 5, 19, 20, 15, 22, 900000)
    assert event_info["region"] == "TURKEY"
    assert event_info["magnitude"] == 5.9
    assert event_info["magnitude_type"] == "Mwc"


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
        project.get_station_filename("HL", "ARG", "", "BHZ", "datalessSEED")
    existing_seed_filename = glob.glob(os.path.join(
        project.paths["dataless_seed"], "dataless.HL_*"))[0]

    assert os.path.exists(existing_seed_filename)
    assert existing_seed_filename != new_seed_filename
    assert os.path.dirname(existing_seed_filename) == \
        os.path.dirname(new_seed_filename)
    assert os.path.dirname(new_seed_filename) == project.paths["dataless_seed"]

    # Test RESP file name generation.
    resp_filename_1 = project.get_station_filename("A", "B", "C", "D", "RESP")
    assert not os.path.exists(resp_filename_1)
    assert os.path.dirname(resp_filename_1) == project.paths["resp"]
    with open(resp_filename_1, "wt") as fh:
        fh.write("blub")
    assert os.path.exists(resp_filename_1)

    resp_filename_2 = project.get_station_filename("A", "B", "C", "D", "RESP")
    assert resp_filename_1 != resp_filename_2
    assert os.path.dirname(resp_filename_2) == project.paths["resp"]


def test_generating_new_iteration(project):
    """
    Tests that iteration creation works.
    """
    assert os.listdir(project.paths["iterations"]) == []

    # Using an invalid solver raises.
    with pytest.raises(LASIFException) as excinfo:
        project.create_new_iteration("1", "unknown_solver")
    msg = excinfo.value.message
    assert "not known" in msg
    assert "unknown_solver" in msg

    # Nothing should have happened.
    assert os.listdir(project.paths["iterations"]) == []

    # Now actually create a new iteration.
    project.create_new_iteration("1", "ses3d_4_0")
    assert os.listdir(project.paths["iterations"]) == ["ITERATION_1.xml"]

    # Creating an already existing iteration raises.
    with pytest.raises(LASIFException) as excinfo:
        project.create_new_iteration("1", "ses3d_4_0")
    assert excinfo.value.message.lower() == "iteration already exists."


def test_iteration_handling(project):
    """
    Tests the managing of the iterations.
    """
    # First create two iterations.
    project.create_new_iteration("1", "ses3d_4_0")
    project.create_new_iteration("2", "ses3d_4_0")
    assert sorted(os.listdir(project.paths["iterations"])) == \
        sorted(["ITERATION_1.xml", "ITERATION_2.xml"])

    # Make sure they are found correctly.
    assert project.get_iteration_dict() == {key: os.path.join(
        project.paths["iterations"], "ITERATION_" + key + ".xml")
        for key in ["1", "2"]}

    iteration = project._get_iteration("1")

    # Assert that the aspects of the example project did get picked up by the
    # iteration.
    assert len(iteration.events) == 2

    assert len(iteration.events["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"]
               ["stations"]) == 4
    assert len(iteration.events["GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"]
               ["stations"]) == 0
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
    project.create_new_iteration("1", "ses3d_4_0")
    processing_tag = project._get_iteration("1").get_processing_tag()
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


def test_single_event_plot(project):
    """
    Tests the plotting of a single event.
    """
    project.plot_event("GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
                       show_plot=False)
    images_are_identical("single_event_plot", project.paths["root"])


def test_simple_raydensity(project):
    """
    Tests the plotting of a single event.
    """
    project.plot_raydensity(show_plot=False, save_plot=False)
    # Use a low dpi to keep the test filesize in check.
    images_are_identical("simple_raydensity_plot", project.paths["root"],
                         dpi=25)


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
    project.create_new_iteration("1", "ses3d_4_0")

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
    def reset():
        project._seismic_events.clear()
        obspy.core.event.ResourceIdentifier\
            ._ResourceIdentifier__resource_id_weak_dict.clear()

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
            "smi:local/76499e98-9ac4-4de1-844b-4042d0e80915",
            "smi:service.iris.edu/fdsnws/event/1/query?eventid=2847365"))
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
    project.create_new_iteration("1", "ses3d_4_0")
    project.preprocess_data("1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
                            waiting_time=0.0)
    iterator = project.data_synthetic_iterator(
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", "1")

    # The example project only contains synthetics for two stations.
    expected = {
        "HL.ARG..BHZ": {"latitude": 36.216, "local_depth": 0.0,
                        "elevation": 170.0, "longitude": 28.126},
        "HT.SIGR..HHZ": {"latitude": 39.2114, "local_depth": 0.0,
                         "elevation": 93.0, "longitude": 25.8553}}

    found = []
    no_synthetics_found_count = 0

    assert recwarn.list == []

    for _i in iterator:
        if _i is None:
            # If the data is not found it should always warn.
            w = recwarn.pop(UserWarning)
            assert "Found 0 not 3" in w.message.args[0]
            no_synthetics_found_count += 1
            continue
        data = _i["data"]
        synth = _i["synthetics"]
        coods = _i["coordinates"]
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
    print(project)
    out = capsys.readouterr()[0]
    assert "\"ExampleProject\"" in out
    assert "Toy Project used in the Test Suite" in out
    assert "2 events" in out
    assert "4 raw waveform files" in out
    assert "0 processed waveform files" in out
    assert "6 synthetic waveform files" in out
