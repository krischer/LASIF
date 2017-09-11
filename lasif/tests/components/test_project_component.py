#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import os

import pathlib
import pytest
import shutil

from lasif.components.project import Project


@pytest.fixture()
def comm(tmpdir):
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "ExampleProject")
    tmpdir = str(tmpdir)
    shutil.copytree(proj_dir, os.path.join(tmpdir, "proj"))
    proj_dir = os.path.join(tmpdir, "proj")
    folder_path = pathlib.Path(proj_dir).absolute()
    project = Project(project_root_path=folder_path, init_project=False)

    return project.comm


def test_config_file_creation_and_parsing(tmpdir):
    """
    Tests the creation of a default config file and the reading of the file.
    """
    # Create a new project.
    pr = Project(project_root_path=pathlib.Path(str(tmpdir)).absolute(),
                 init_project="TestProject")
    del pr

    # Init it once again.
    pr = Project(pathlib.Path(str(tmpdir)).absolute())

    # Assert the config file will test the creation of the default file and the
    # reading.
    assert pr.config["project_name"] == "TestProject"
    assert pr.config["description"] == ""
    assert pr.config["download_settings"]["channel_priorities"] == [
        "BH[Z,N,E]", "LH[Z,N,E]", "HH[Z,N,E]", "EH[Z,N,E]", "MH[Z,N,E]"]
    assert pr.config["download_settings"]["location_priorities"] == [
        "", "00", "10", "20", "01", "02"]
    assert pr.config["download_settings"]["interstation_distance_in_meters"] == \
        1000.0
    assert pr.config["download_settings"]["seconds_after_event"] == 3600.0
    assert pr.config["download_settings"]["seconds_before_event"] == 300.0


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
    print(out)
    assert "\"ExampleProject\"" in out
    assert "Toy Project used in the Test Suite" in out
    assert "1 events" in out


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
