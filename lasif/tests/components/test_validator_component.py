#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import os
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

    project = Project(project_root_path=proj_dir, init_project=False)

    return project.comm


def test_is_event_station_raypath_within_boundaries(comm):
    """
    Tests the raypath checker.
    """
    # latitude = 38.82
    # longitude = 40.14
    event = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"

    assert comm.validator.is_event_station_raypath_within_boundaries(
        event, 38.92, 40.0)
    assert not comm.validator.is_event_station_raypath_within_boundaries(
        event, 38.92, 140.0)
