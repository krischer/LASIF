#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for some utility functions.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import numpy as np
from obspy import UTCDateTime
from obspy.io.xseed import Parser
import os

from lasif import utils

data_dir = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


def test_channel_in_parser():
    """
    Tests if a given channel is part of a Parser object.
    """
    starttime = UTCDateTime(2007, 2, 12, 10, 30, 28, 197700)
    endtime = UTCDateTime(2007, 2, 12, 11, 35, 28, 197700)
    channel_id = "ES.ECAL..HHE"
    # An empty file should of course not contain much.
    parser_object = Parser(os.path.join(data_dir, "station_files", "seed",
                           "channelless_datalessSEED"))
    assert utils.channel_in_parser(parser_object, channel_id,
                                   starttime, endtime) is False
    # Now read a file that actually contains data.
    channel_id = "IU.PAB.00.BHE"
    starttime = UTCDateTime(1999, 2, 18, 10, 0)
    endtime = UTCDateTime(2009, 8, 13, 19, 0)
    parser_object = Parser(os.path.join(data_dir, "station_files", "seed",
                           "dataless.IU_PAB"))
    # This is an exact fit of the start and end times in this file.
    assert utils.channel_in_parser(
        parser_object, channel_id, starttime, endtime) is True
    # Now try some others that do not fit.
    assert utils.channel_in_parser(
        parser_object, channel_id, starttime - 1, endtime) is False
    assert utils.channel_in_parser(
        parser_object, channel_id, starttime, endtime + 1) is False
    assert utils.channel_in_parser(
        parser_object, channel_id + "x", starttime, endtime) is False
    assert utils.channel_in_parser(
        parser_object, channel_id, starttime - 200, starttime - 100) is False
    assert utils.channel_in_parser(
        parser_object, channel_id, endtime + 100, endtime + 200) is False
    # And some that do fit.
    assert utils.channel_in_parser(
        parser_object, channel_id, starttime, starttime + 10) is True
    assert utils.channel_in_parser(
        parser_object, channel_id, endtime - 100, endtime) is True


def test_greatcircle_points_generator():
    """
    Tests the greatcircle discrete point generator.
    """
    points = list(utils.greatcircle_points(
        utils.Point(0, 0), utils.Point(0, 90), max_npts=90))
    assert len(points) == 90
    assert [_i.lat for _i in points] == 90 * [0.0]
    np.testing.assert_array_almost_equal([_i.lng for _i in points],
                                         np.linspace(0, 90, 90))

    points = list(utils.greatcircle_points(
        utils.Point(0, 0), utils.Point(0, 90), max_npts=110))
    assert len(points) == 110
