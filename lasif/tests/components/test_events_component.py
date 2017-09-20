#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import obspy
import os
import pytest
from unittest import mock

from lasif import LASIFNotFoundError
from lasif.components.events import EventsComponent
from lasif.components.communicator import Communicator


@pytest.fixture
def comm():
    """
    Returns a communicator with an initialized events component.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "ExampleProject",
        "DATA/EARTHQUAKES")
    comm = Communicator()
    # Add project comm with paths to this fake component.
    comm.project = mock.MagicMock()
    comm.project.paths = {"root": data_dir}
    EventsComponent(data_dir, comm, "events")
    return comm


def test_event_list(comm):
    print(comm.events.list())
    print("bla")
    assert sorted(comm.events.list()) == sorted([
        "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10",
        "GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_5.8_2014-10-15-13"])


def test_event_count(comm):
    assert comm.events.count() == 2


def test_has_event(comm):
    assert comm.events.has_event(
        "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10")
    assert comm.events.has_event(
        "GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_5.8_2014-10-15-13")
    assert not comm.events.has_event("random")


def test_get_event(comm):
    ev = comm.events.get("GCMT_event_ICELAND_Mag_5.5_2014-10-7-10")
    print(ev)
    ev["filename"] = os.path.basename(ev["filename"])
    assert ev == {
        'depth_in_km': 12.0,
        'event_name': 'GCMT_event_ICELAND_Mag_5.5_2014-10-7-10',
        'filename': 'GCMT_event_ICELAND_Mag_5.5_2014-10-7-10.h5',
        'latitude': 64.62,
        'longitude': -17.26,
        'm_pp': 1.23e+17,
        'm_rp': -7.74e+16,
        'm_rr': -2.97e+17,
        'm_rt': 3.87e+16,
        'm_tp': -1900000000000000.0,
        'm_tt': 1.74e+17,
        'magnitude': 5.53,
        'magnitude_type': 'Mwc',
        'origin_time': obspy.UTCDateTime(2014, 10, 7, 10, 22, 34, 100000),
        'region': u'ICELAND'}

    ev = comm.events.get("GCMT_event_IRAN-IRAQ_BORDER_"
                         "REGION_Mag_5.8_2014-10-15-13")
    ev["filename"] = os.path.basename(ev["filename"])
    assert ev == {
        'depth_in_km': 12.0,
        'event_name':
            'GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_5.8_2014-10-15-13',
        'filename':
            'GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_5.8_2014-10-15-13.h5',
        'latitude': 32.45,
        'longitude': 47.87,
        'm_pp': 4.36e+16,
        'm_rp': 1.65e+16,
        'm_rr': 5.81e+17,
        'm_rt': -1.76e+17,
        'm_tp': 7000000000000000.0,
        'm_tt': -6.25e+17,
        'magnitude': 5.8,
        'magnitude_type': 'Mwc',
        'origin_time': obspy.UTCDateTime(2014, 10, 15, 13, 35, 57, 900000),
        'region': u'IRAN-IRAQ BORDER REGION'}

    with pytest.raises(LASIFNotFoundError):
        comm.events.get("random")


def test_get_all_events(comm):
    events = {
        "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10":
            comm.events.get("GCMT_event_ICELAND_Mag_5.5_2014-10-7-10"),
        "GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_5.8_2014-10-15-13":
            comm.events.get("GCMT_event_IRAN-IRAQ_"
                            "BORDER_REGION_Mag_5.8_2014-10-15-13")}
    assert events == comm.events.get_all_events()
