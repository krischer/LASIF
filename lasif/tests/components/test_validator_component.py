#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import mock
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


def test_data_validation_raypath_in_domain(comm):
    """
    Tests the raypath in domain checker and associated output.
    """
    # Initially everything is ok.
    assert comm.validator.validate_raypaths_in_domain() == []
    event = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    folder = os.path.join(comm.project.paths["data"], event, "raw")
    filenames = sorted([os.path.join(folder, _i) for _i in os.listdir(folder)])

    # Have the raypath check fail.
    with mock.patch('lasif.components.validator.ValidatorComponent'
                    '.is_event_station_raypath_within_boundaries') as p:
        p.return_value = False
        assert sorted(comm.validator.validate_raypaths_in_domain()) == \
            filenames
        # 4 stations, 4 calls
        assert p.call_count == 4


def test_data_validation(comm, capsys):
    """
    Attempt to test the data validation part in a simple manner.
    """
    # The default output should be fully valid.
    comm.validator.validate_data()
    out = capsys.readouterr()[0]
    assert "ALL CHECKS PASSED" in out

    filename = os.path.join(comm.project.paths["events"],
                            "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11.xml")
    with open(filename, "rt") as fh:
        original_contents = fh.read()

    # Now make a faulty QuakeML file. Removing a public id will trigger an
    # error int he QuakeML validation.
    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            'publicID="smi:local/76499e98-9ac4-4de1-844b-4042d0e80915"', ""))
    comm.validator.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Validating against ")][0]
    assert "FAIL" in line

    # Now duplicate an id.
    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            "smi:service.iris.edu/fdsnws/event/1/query?eventid=2847365",
            "smi:www.iris.edu/spudservice/momenttensor/gcmtid/"
            "C201003241411A#reforigin"))
    comm.validator.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Checking for duplicate ")][0]
    assert "FAIL" in line

    # Now make the file have an insane depth. This should trigger a sanity
    # check error.
    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            "<value>4500.0</value>",
            "<value>450000000000.0</value>"))
    comm.validator.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Performing some basic sanity ")][0]
    assert "FAIL" in line

    # Trigger an error that two events are too close in time.
    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            "2010-03-24T14:11:31.000000Z",
            "2011-05-19T20:15:22.900000Z"))
    comm.events.update_cache()
    comm.validator.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Checking for duplicates and ")][0]
    assert "FAIL" in line

    # Create an event outside of the chosen domain.

    # One either has to wait for a bit or delete the current event cache
    # manually. We'll just delete the cache...
    c_filename = os.path.join(comm.project.paths["cache"],
                              "event_cache.sqlite")
    try:
        os.remove(c_filename)
    except:
        pass

    with open(filename, "wt") as fh:
        fh.write(original_contents.replace(
            "<value>40.14</value>",
            "<value>-100.0</value>"))
    comm.events.update_cache()
    comm.validator.validate_data()
    out = capsys.readouterr()[0]
    line = [_i.strip() for _i in out.split("\n")
            if _i.strip().startswith("Assure all events are in chosen")][0]
    assert "FAIL" in line
