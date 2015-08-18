#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import inspect
import os
import pytest
import shutil

from lasif.domain import RectangularSphericalSection
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


def test_config_file_creation_and_parsing(tmpdir):
    """
    Tests the creation of a default config file and the reading of the file.
    """
    # Create a new project.
    pr = Project(project_root_path=str(tmpdir), init_project="TestProject")
    del pr

    # Init it once again.
    pr = Project(str(tmpdir))

    # Assert the config file will test the creation of the default file and the
    # reading.
    assert pr.config["name"] == "TestProject"
    assert pr.config["description"] == ""
    assert pr.config["download_settings"]["channel_priorities"] == [
        "BH[Z,N,E]", "LH[Z,N,E]", "HH[Z,N,E]", "EH[Z,N,E]", "MH[Z,N,E]"]
    assert pr.config["download_settings"]["location_priorities"] == [
        "", "00", "10", "20", "01", "02"]
    assert pr.config["download_settings"]["interstation_distance_in_m"] == \
        1000.0
    assert pr.config["download_settings"]["seconds_after_event"] == 3600.0
    assert pr.config["download_settings"]["seconds_before_event"] == 300.0

    d = RectangularSphericalSection(
        min_latitude=-20.0, max_latitude=20.0, min_longitude=-20.0,
        max_longitude=20.0, min_depth_in_km=0.0, max_depth_in_km=200.0,
        rotation_angle_in_degree=-45.0,
        rotation_axis=[1.0, 1.0, 1.0],
        boundary_width_in_degree=3.0)
    assert pr.domain == d


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


def test_output_folder_name(comm):
    """
    Silly test to safeguard against regressions.
    """
    import obspy
    cur_time = obspy.UTCDateTime()

    output_dir = comm.project.get_output_folder(
        type="random", tag="some_string")

    basename = os.path.basename(output_dir)
    type_dir = os.path.basename(os.path.dirname(output_dir))
    assert type_dir == "random"

    time, tag = basename.split("__")
    time = obspy.UTCDateTime(time)

    assert (time - cur_time) <= 1.0
    assert tag == "some_string"


def test_string_representation(comm, capsys):
    """
    Tests the projects string representation.
    """
    print(comm.project)
    out = capsys.readouterr()[0]
    assert "\"ExampleProject\"" in out
    assert "Toy Project used in the Test Suite" in out
    assert "2 events" in out
    assert "4 station files" in out
    assert "6 raw waveform files" in out
    assert "0 processed waveform files" in out
    assert "6 synthetic waveform files" in out


def test_log_filename_creation(comm):
    """
    Tests the logfiles.
    """
    import obspy
    cur_time = obspy.UTCDateTime()

    log_file = comm.project.get_log_file("DOWNLOADS", "some_event")
    log_dir = os.path.dirname(log_file)
    assert log_dir == os.path.join(comm.project.paths["logs"], "DOWNLOADS")

    basename = os.path.basename(log_file)
    time, desc = basename.split("___")
    time = obspy.UTCDateTime(time)

    assert (time - cur_time) <= 0.1
    assert desc == "some_event.log"
