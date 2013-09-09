#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test cases for the project class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import copy
import inspect
import matplotlib.pylab as plt
import obspy
import os
import pytest
import shutil
import time

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


def images_are_identical(expected, actual):
    """
    Partially copied from ObsPy
    """
    from matplotlib.testing.compare import compare_images as mpl_compare_images
    from matplotlib.pyplot import rcdefaults
    # set matplotlib builtin default settings for testing
    rcdefaults()
    import locale
    locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))
    if mpl_compare_images(expected, actual, 0.001) is None:
        return True
    else:
        return False


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

    a = time.time()
    pr = Project(str(tmpdir), init_project="TestProject")
    b = time.time()
    time_for_non_cached_read = b - a

    # Assert that everything is still the same.
    assert config == pr.config
    assert domain == pr.domain
    del pr

    # This should have created the cached file.
    assert os.path.exists(cache)

    a = time.time()
    pr = Project(str(tmpdir), init_project="TestProject")
    b = time.time()
    time_for_cached_read = b - a

    # A cached read should always be faster.
    assert time_for_cached_read < time_for_non_cached_read

    # Assert that nothing changed.
    assert config == pr.config
    assert domain == pr.domain


def test_domain_plotting(tmpdir):
    """
    Very simple domain plotting test.
    """
    pr = Project(str(tmpdir), init_project="TestProject")
    pr.plot_domain(show_plot=False)

    baseline_image = os.path.join(IMAGES, "simple_test_domain.png")
    this_image = os.path.join(str(tmpdir), "simple_test_domain.png")

    plt.savefig(this_image)
    plt.close()

    assert images_are_identical(baseline_image, this_image)


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
    baseline_image = os.path.join(IMAGES, "two_events_plot_map.png")
    this_image = os.path.join(project.paths["root"], "two_events_plot_map.png")
    project.plot_events(plot_type="map", show_plot=False)
    plt.savefig(this_image)
    assert images_are_identical(baseline_image, this_image)
    plt.close()

    baseline_image = os.path.join(IMAGES, "two_events_plot_depth.png")
    this_image = os.path.join(project.paths["root"],
                              "two_events_plot_depth.png")
    project.plot_events(plot_type="depth", show_plot=False)
    plt.savefig(this_image)
    assert images_are_identical(baseline_image, this_image)
    plt.close()

    baseline_image = os.path.join(IMAGES, "two_events_plot_time.png")
    this_image = os.path.join(project.paths["root"],
                              "two_events_plot_time.png")
    project.plot_events(plot_type="time", show_plot=False)
    plt.savefig(this_image)
    assert images_are_identical(baseline_image, this_image)
    plt.close()


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
    a = time.time()
    cache = project._get_waveform_cache_file(event_name, "raw",
                                             show_progress=False)
    b = time.time()
    time_for_uncached_cache_retrieval = b - a

    # Make sure it now exists.
    assert os.path.exists(waveform_cache)

    # Accessing it again should be much faster as it already exists.
    a = time.time()
    cache = project._get_waveform_cache_file(event_name, "raw",
                                             show_progress=False)
    b = time.time()
    time_for_cached_cache_retrieval = b - a

    assert time_for_cached_cache_retrieval < time_for_uncached_cache_retrieval

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

