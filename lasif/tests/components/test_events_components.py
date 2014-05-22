#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import obspy
import os
import pytest
import time

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
        "EVENTS")
    comm = Communicator()
    EventsComponent(data_dir, comm, "events")
    return comm


def test_event_list(comm):
    assert sorted(comm.events.list()) == sorted([
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
        "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"])


def test_event_count(comm):
    assert comm.events.count() == 2


def test_has_event(comm):
    assert comm.events.has_event(
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    assert comm.events.has_event(
        "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15")
    assert not comm.events.has_event("random")


def test_get_event(comm):
    ev = comm.events.get("GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    ev["filename"] = os.path.basename(ev["filename"])
    assert ev == {
        'depth_in_km': 4.5,
        'event_name': 'GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11',
        'filename': 'GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11.xml',
        'latitude': 38.82,
        'longitude': 40.14,
        'm_pp': 3.56e+16,
        'm_rp': -2.25e+16,
        'm_rr': 5470000000000000.0,
        'm_rt': 2.26e+16,
        'm_tp': 1.92e+16,
        'm_tt': -4.11e+16,
        'magnitude': 5.1,
        'magnitude_type': 'Mwc',
        'origin_time': obspy.UTCDateTime(2010, 3, 24, 14, 11, 31),
        'region': u'TURKEY'}

    ev = comm.events.get("GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15")
    ev["filename"] = os.path.basename(ev["filename"])
    assert ev == {
        'depth_in_km': 7.0,
        'event_name': 'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15',
        'filename': 'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15.xml',
        'latitude': 39.15,
        'longitude': 29.1,
        'm_pp': -8.5e+16,
        'm_rp': -5.3e+16,
        'm_rr': -8.07e+17,
        'm_rt': 2.8e+16,
        'm_tp': -2.17e+17,
        'm_tt': 8.92e+17,
        'magnitude': 5.9,
        'magnitude_type': 'Mwc',
        'origin_time': obspy.UTCDateTime(2011, 5, 19, 20, 15, 22, 900000),
        'region': u'TURKEY'}

    with pytest.raises(LASIFNotFoundError):
        comm.events.get("random")


def test_get_all_events(comm):
    events = {
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11":
            comm.events.get("GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"),
        "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15":
            comm.events.get("GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15")}
    assert events == comm.events.get_all_events()


def test_event_caching(comm):
    """
    Tests that the caching actually does something. In my tests the caching
    results in a speedup of 1000 so testing for a 10 times speedup should be
    ok. If not, remove this test.
    """
    a = time.time()
    comm.events.get_all_events()
    b = time.time()
    first_run = b - a

    a = time.time()
    comm.events.get_all_events()
    b = time.time()
    second_run = b - a

    assert (second_run * 10) <= first_run
