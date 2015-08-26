#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the windows component.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import mock
import numpy as np
import os
import pytest

from lasif import LASIFNotFoundError
from lasif.window_manager import WindowGroupManager
from ..testing_helpers import communicator  # NOQA


@pytest.fixture
def comm(communicator):
    with mock.patch("lasif.tools.Q_discrete.calculate_Q_model") as patch:
        # Speed up this test.
        patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                              np.array([0.59496, 3.7119, 22.2171]))

        communicator.iterations.create_new_iteration(
            iteration_name=1,
            solver_name="SES3D_4_1",
            events_dict=communicator.query.get_stations_for_all_events(),
            min_period=40.0,
            max_period=100.0)

    # Create folders for each event.
    for event_name in communicator.events.list():
        os.makedirs(os.path.join(communicator.project.paths["windows"],
                                 event_name))

    # Also create an iteration for one of the events.
    first_event = sorted(communicator.events.list())[0]
    os.makedirs(os.path.join(
        communicator.project.paths["windows"], first_event,
        communicator.iterations.get_long_iteration_name("1")))

    return communicator


def test_window_list(comm):
    """
    Tests the list() method.
    """
    assert comm.windows.list() == sorted([
        'GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11',
        'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15'])


def test_window_list_for_event(comm):
    """
    Tests the list_for_event() method.
    """
    assert comm.windows.list_for_event(
        'GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11') == ["1"]
    assert comm.windows.list_for_event(
        'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15') == []


def test_get(comm):
    """
    Tests the get() method.
    """
    # No known event.
    with pytest.raises(LASIFNotFoundError) as error:
        comm.windows.get("RANDOM_EVENT", "RANDOM_ITERATION")
    assert "event" in str(error.value).lower()
    assert "not known" in str(error.value).lower()

    # Known event, but unknown iteration.
    with pytest.raises(LASIFNotFoundError) as error:
        comm.windows.get('GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11',
                         'RANDOM_ITERATION')
    assert "iteration" in str(error.value).lower()
    assert "not found" in str(error.value).lower()

    wm = comm.windows.get('GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11', '1')
    assert isinstance(wm, WindowGroupManager)
