#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import os
import pytest
import shutil

from lasif import LASIFError, LASIFNotFoundError
from lasif.components.project import Project
from ..testing_helpers import DATA


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


def test_get_all_stations_for_event(comm):
    """
    Tests the get_all_stations_for_event method.
    """
    event_1 = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    event_2 = "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"

    # Get all stations for event_1.
    stations_1 = comm.query.get_all_stations_for_event(event_1)
    assert len(stations_1) == 4
    assert sorted(stations_1.keys()) == sorted(["HL.ARG", "HT.SIGR", "KO.KULA",
                                                "KO.RSDY"])
    assert stations_1["HL.ARG"] == {
        "latitude": 36.216, "local_depth_in_m": 0.0, "elevation_in_m": 170.0,
        "longitude": 28.126}

    # event_2 has no stations.
    with pytest.raises(LASIFNotFoundError):
        comm.query.get_all_stations_for_event(event_2)


def test_get_coordinates_for_station(comm):
    """
    Tests the get_coordiantes_for_station() method.
    """
    event = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    station = comm.query.get_coordinates_for_station(event,
                                                     station_id="HL.ARG")
    assert station == {"latitude": 36.216, "local_depth_in_m": 0.0,
                       "elevation_in_m": 170.0, "longitude": 28.126}


def test_get_debug_information_for_file(comm):
    """
    Test the what_is() method.
    """
    # Test for SEED files
    info = comm.query.what_is(os.path.join(
        comm.project.paths["dataless_seed"], "dataless.HL_ARG"))
    assert info == (
        "The SEED file contains information about 3 channels:\n"
        "\tHL.ARG..BHE | 2003-03-04T00:00:00.000000Z - -- | Lat/Lng/Ele/Dep: "
        "36.22/28.13/170.00/0.00\n"
        "\tHL.ARG..BHN | 2003-03-04T00:00:00.000000Z - -- | Lat/Lng/Ele/Dep: "
        "36.22/28.13/170.00/0.00\n"
        "\tHL.ARG..BHZ | 2003-03-04T00:00:00.000000Z - -- | Lat/Lng/Ele/Dep: "
        "36.22/28.13/170.00/0.00")

    # Test a RESP file.
    resp_file = os.path.join(comm.project.paths["resp"], "RESP.AF.DODT..BHE")
    shutil.copy(os.path.join(DATA, "station_files", "resp",
                             "RESP.AF.DODT..BHE"), resp_file)
    # Manually reload the station cache.
    comm.stations.force_cache_update()
    info = comm.query.what_is(resp_file)
    assert info == (
        "The RESP file contains information about 1 channel:\n"
        "\tAF.DODT..BHE | 2009-08-09T00:00:00.000000Z - -- | "
        "Lat/Lng/Ele/Dep: --/--/--/--")

    # Any other file should simply return an error.
    with pytest.raises(LASIFError) as excinfo:
        comm.query.what_is(os.path.join(DATA, "File_r"))
    assert "is not part of the LASIF project." in excinfo.value.message
    with pytest.raises(LASIFNotFoundError) as excinfo:
        comm.query.what_is(os.path.join("random", "path"))
    assert "does not exist" in excinfo.value.message

    # Test a MiniSEED file.
    info = comm.query.what_is(os.path.join(
        comm.project.paths["data"],
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
        "raw", "HL.ARG..BHZ.mseed"))
    assert info == (
        "The MSEED file contains 1 channel:\n"
        "	HL.ARG..BHZ | 2010-03-24T14:06:31.024999Z - "
        "2010-03-24T15:11:30.974999Z | Lat/Lng/Ele/Dep: --/--/--/--")

    # Testing a SAC file.
    sac_file = os.path.join(
        comm.project.paths["data"],
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
        "raw", "CA.CAVN..HHN.SAC_cut")
    shutil.copy(os.path.join(DATA, "CA.CAVN..HHN.SAC_cut"), sac_file)
    info = comm.query.what_is(sac_file)
    assert info == (
        "The SAC file contains 1 channel:\n"
        "	CA.CAVN..HHN | 2008-02-20T18:28:02.997002Z - "
        "2008-02-20T18:28:04.997002Z | Lat/Lng/Ele/Dep: "
        "41.88/0.75/634.00/0.00")
