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
