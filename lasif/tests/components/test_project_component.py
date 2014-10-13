#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

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
