#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the event pseudo dictionary.

:copyright: Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license: GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import itertools
import obspy
import os
import pytest
import types

from lasif.tools.event_pseudo_dict import EventPseudoDict
from lasif.tests.testing_helpers import DATA


@pytest.fixture
def event_pseudo_dict():
    """
    Fixture creating a fresh EventPseudoDict pointing to the events folder in
    the example project. Thus it will contain two events.
    """
    ev = EventPseudoDict(os.path.join(DATA, "ExampleProject", "EVENTS"))
    return ev

# Define both events, as they do not change.
EVENT_1_NAME = "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"
EVENT_1_INFO = {
    "event_name": "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15",
    "latitude": 39.15,
    "longitude": 29.1,
    "origin_time": obspy.UTCDateTime(2011, 5, 19, 20, 15, 22, 900000),
    "depth_in_km": 7.0,
    "magnitude": 5.9,
    "magnitude_type": "Mwc",
    "region": "TURKEY",
    "m_rr": -8.07e+17,
    "m_tt": 8.92e+17,
    "m_pp": -8.5e+16,
    "m_rt": 2.8e+16,
    "m_rp": -5.3e+16,
    "m_tp": -2.17e+17}
EVENT_2_NAME = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
EVENT_2_INFO = {
    "event_name": "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
    "latitude": 38.82,
    "longitude": 40.14,
    "origin_time": obspy.UTCDateTime(2010, 3, 24, 14, 11, 31),
    "depth_in_km": 4.5,
    "magnitude": 5.1,
    "magnitude_type": "Mwc",
    "region": "TURKEY",
    "m_rr": 5470000000000000.0,
    "m_tt": -4.11e+16,
    "m_pp": 3.56e+16,
    "m_rt": 2.26e+16,
    "m_rp": -2.25e+16,
    "m_tp": 1.92e+16}


def test_keys(event_pseudo_dict):
    """
    Tests the keys() method.
    """
    keys = event_pseudo_dict.keys()
    assert sorted(keys) == sorted([EVENT_1_NAME, EVENT_2_NAME])
    assert isinstance(keys, list)


def test_iterkeys(event_pseudo_dict):
    """
    Tests the iterkeys() method.
    """
    iterkeys = event_pseudo_dict.iterkeys()
    assert sorted(iterkeys) == sorted([EVENT_1_NAME, EVENT_2_NAME])
    assert isinstance(iterkeys, types.GeneratorType)


def test_values(event_pseudo_dict):
    """
    Tests the values() method.
    """
    values = event_pseudo_dict.values()
    assert sorted(values) == sorted([EVENT_1_INFO, EVENT_2_INFO])
    assert isinstance(values, list)


def test_itervalues(event_pseudo_dict):
    """
    Tests the itervalues() method.
    """
    itervalues = event_pseudo_dict.itervalues()
    assert sorted(itervalues) == sorted([EVENT_1_INFO, EVENT_2_INFO])
    assert isinstance(itervalues, types.GeneratorType)


def test_items(event_pseudo_dict):
    """
    Tests the items() method.
    """
    items = event_pseudo_dict.items()
    assert sorted(items) == sorted([(EVENT_1_NAME, EVENT_1_INFO),
                                    (EVENT_2_NAME, EVENT_2_INFO)])
    assert isinstance(items, list)


def test_iteritems(event_pseudo_dict):
    """
    Tests the iteritems() method.
    """
    iteritems = event_pseudo_dict.iteritems()
    assert sorted(iteritems) == sorted([(EVENT_1_NAME, EVENT_1_INFO),
                                        (EVENT_2_NAME, EVENT_2_INFO)])
    # Iteritems return an itertools.izip instance which for some reason is not
    # considered a generator. But it is an iterator notheless.
    assert isinstance(iteritems, itertools.izip)


def test_contains(event_pseudo_dict):
    """
    Tests the __contains__ special method.
    """
    assert EVENT_1_NAME in event_pseudo_dict
    assert EVENT_2_NAME in event_pseudo_dict
    assert not "test" in event_pseudo_dict
    assert not 1 in event_pseudo_dict


def test_len(event_pseudo_dict):
    """
    Tests the __len__ special method.
    """
    assert len(event_pseudo_dict) == 2


def test_getitem(event_pseudo_dict):
    """
    Tests the __getitem__ special method.
    """
    assert event_pseudo_dict[EVENT_1_NAME] == EVENT_1_INFO
    assert event_pseudo_dict[EVENT_2_NAME] == EVENT_2_INFO

    with pytest.raises(KeyError):
        event_pseudo_dict["test"]

    with pytest.raises(KeyError):
        event_pseudo_dict[1]
