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


def test_waveform_cache_usage(comm):
    """
    Tests the automatic creation and usage of the waveform caches.
    """
    event_name = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    waveform_cache = os.path.join(comm.project.paths["data"], event_name,
                                  "raw_cache.sqlite")
    waveform_folder = os.path.join(comm.project.paths["data"], event_name,
                                   "raw")

    # The example project does not yet have the cache.
    assert not os.path.exists(waveform_cache)

    # Create the cache.
    cache = comm.waveforms.get_waveform_cache(event_name, "raw")

    # Make sure it now exists.
    assert os.path.exists(waveform_cache)

    # The cache has to point to the correct folder and file.
    assert cache.waveform_folder == waveform_folder
    assert cache.cache_db_file == waveform_cache
    # Make sure the cache contains all files. The files attribute contains
    # relative paths inside the cache.
    assert sorted([os.path.join(cache.root_folder, _i) for _i in
                   cache.files["waveform"]]) == \
        sorted([os.path.join(waveform_folder, _i)
                for _i in os.listdir(waveform_folder)])

    # Tests an exemplary file.
    filename = os.path.join(comm.project.paths["data"], event_name, "raw",
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
