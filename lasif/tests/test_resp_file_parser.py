#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test suite for the RESP file parser.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import obspy
import os


# Most generic way to get the actual data directory.
from lasif.file_handling import simple_resp_parser

data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data", "station_files", "resp")


def test_reading_simple_resp_file():
    """
    Tests reading of a very simple RESP file.
    """
    filename = os.path.join(data_dir, "RESP.G.FDF.00.BHE")
    channels = simple_resp_parser.get_inventory(filename)
    assert len(channels) == 1
    channel = channels[0]
    assert channel["network"] == "G"
    assert channel["station"] == "FDF"
    assert channel["location"] == "00"
    assert channel["channel"] == "BHE"
    assert channel["start_date"] == obspy.UTCDateTime(2006, 11, 22, 13)
    assert channel["end_date"] is None
    assert channel["channel_id"] == "G.FDF.00.BHE"


def test_reading_more_complex_resp_file():
    """
    Tests reading a slightly more complex RESP file. This one contains two
    (identical!!!) channels. That is pretty much the maximum complexity of
    RESP files.
    """
    filename = os.path.join(data_dir, "RESP.AF.DODT..BHE")
    channels = simple_resp_parser.get_inventory(filename)
    assert len(channels) == 2
    channel = channels[0]
    assert channel["network"] == "AF"
    assert channel["station"] == "DODT"
    assert channel["location"] == ""
    assert channel["channel"] == "BHE"
    assert channel["start_date"] == obspy.UTCDateTime(2009, 8, 9, 0)
    assert channel["end_date"] is None
    assert channel["channel_id"] == "AF.DODT..BHE"
    # The two have the same information for some reason...
    channel = channels[1]
    assert channel["network"] == "AF"
    assert channel["station"] == "DODT"
    assert channel["location"] == ""
    assert channel["channel"] == "BHE"
    assert channel["start_date"] == obspy.UTCDateTime(2009, 8, 9, 0)
    assert channel["end_date"] is None
    assert channel["channel_id"] == "AF.DODT..BHE"


def test_removing_duplicates():
    """
    Tests the removal of duplicates.
    """
    filename = os.path.join(data_dir, "RESP.AF.DODT..BHE")
    channels = simple_resp_parser.get_inventory(filename,
                                                remove_duplicates=True)
    assert len(channels) == 1
    channel = channels[0]
    assert channel["network"] == "AF"
    assert channel["station"] == "DODT"
    assert channel["location"] == ""
    assert channel["channel"] == "BHE"
    assert channel["start_date"] == obspy.UTCDateTime(2009, 8, 9, 0)
    assert channel["end_date"] is None
    assert channel["channel_id"] == "AF.DODT..BHE"
