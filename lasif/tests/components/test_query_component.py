#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import os
import pytest
import shutil

from lasif import LASIFError
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


def test_discover_available_data(comm):
    """
    Tests the discover available data method.
    """
    event = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"

    # At the beginning it contains nothing, except a raw vertical component
    assert comm.query.discover_available_data(event, "HL.ARG") == \
        {"processed": {}, "synthetic": {}, "raw": {"raw": ["Z"]}}

    # Create a new iteration. At this point it should contain some synthetics.
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    assert comm.query.discover_available_data(event, "HL.ARG") == \
        {"processed": {},
         "synthetic": {"1": ["Z", "N", "E"]},
         "raw": {"raw": ["Z"]}}

    # A new iteration without data does not add anything.
    comm.iterations.create_new_iteration(
        "2", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    assert comm.query.discover_available_data(event, "HL.ARG") == \
        {"processed": {},
         "synthetic": {"1": ["Z", "N", "E"]},
         "raw": {"raw": ["Z"]}}

    # Data is also available for a second station. But not for another one.
    assert comm.query.discover_available_data(event, "HT.SIGR") == \
        {"processed": {},
         "synthetic": {"1": ["Z", "N", "E"]},
         "raw": {"raw": ["Z"]}}
    assert comm.query.discover_available_data(event, "KO.KULA") == \
        {"processed": {},
         "synthetic": {},
         "raw": {"raw": ["Z"]}}

    # Requesting data for a non-existent station raises.
    with pytest.raises(LASIFError):
        comm.query.discover_available_data(event, "NET.STA")

    # Now preprocess some data that then should appear.
    processing_tag = comm.iterations.get("1").processing_tag
    comm.actions.preprocess_data("1", [event], waiting_time=0.0)
    assert comm.query.discover_available_data(event, "HT.SIGR") == \
        {"processed": {processing_tag: ["Z"]},
         "synthetic": {"1": ["Z", "N", "E"]},
         "raw": {"raw": ["Z"]}}
    assert comm.query.discover_available_data(event, "KO.KULA") == \
        {"processed": {processing_tag: ["Z"]},
         "synthetic": {},
         "raw": {"raw": ["Z"]}}
