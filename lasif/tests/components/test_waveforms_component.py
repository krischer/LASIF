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


def test_get_metadata_synthetic(comm):
    """
    Tests the get metadata synthetic function.
    """
    ev = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)
    meta = comm.waveforms.get_metadata_synthetic(ev, "1")
    channel_ids = sorted([_i["channel_id"] for _i in meta])
    assert channel_ids == sorted([
        "HL.ARG..X",
        "HL.ARG..Y",
        "HL.ARG..Z",
        "HT.SIGR..X",
        "HT.SIGR..Y",
        "HT.SIGR..Z"])


def test_reading_synthetics(comm):
    """
    Tests the reading of synthetic files..
    """
    comm.iterations.create_new_iteration(
        "1", "ses3d_4_1", comm.query.get_stations_for_all_events(), 8, 100)

    event_name = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"

    # Currently no rotation is defined in the project and thus nothing should
    # be rotated.
    with mock.patch("lasif.rotations.rotate_data") as patch:
        st = comm.waveforms.get_waveforms_synthetic(event_name, "HL.ARG", 1)
        assert patch.call_count == 0

    assert len(st) == 3

    assert st[0].id == "HL.ARG..E"
    assert st[1].id == "HL.ARG..N"
    assert st[2].id == "HL.ARG..Z"

    origin_time = comm.events.get(event_name)["origin_time"]
    assert st[0].stats.starttime == origin_time
    assert st[1].stats.starttime == origin_time
    assert st[2].stats.starttime == origin_time

    # Now set the rotation angle. This means that now the synthetics have to be
    # rotated!
    comm.project.domain.rotation_angle_in_degree = 90.0
    with mock.patch("lasif.rotations.rotate_data") as patch:
        patch.return_value = [tr.data for tr in st]
        st = comm.waveforms.get_waveforms_synthetic(event_name, "HL.ARG", 1)
        assert patch.call_count == 1

    # Actually execute it instead of using the mock.
    st = comm.waveforms.get_waveforms_synthetic(event_name, "HL.ARG", 1)

    # The rest should remain the same.
    assert len(st) == 3

    assert st[0].id == "HL.ARG..E"
    assert st[1].id == "HL.ARG..N"
    assert st[2].id == "HL.ARG..Z"

    assert st[0].stats.starttime == origin_time
    assert st[1].stats.starttime == origin_time
    assert st[2].stats.starttime == origin_time

