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
import unittest

from lasif.tools import simple_resp_parser


class RESPFileParserTestCase(unittest.TestCase):
    """
    Simple test case for the RESP file parser.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_reading_simple_resp_file(self):
        """
        Tests reading of a very simple RESP file.
        """
        filename = os.path.join(self.data_dir, "RESP.G.FDF.00.BHE")
        channels = simple_resp_parser.get_inventory(filename)
        self.assertEqual(len(channels), 1)
        channel = channels[0]
        self.assertEqual(channel["network"], "G")
        self.assertEqual(channel["station"], "FDF")
        self.assertEqual(channel["location"], "00")
        self.assertEqual(channel["channel"], "BHE")
        self.assertEqual(channel["start_date"],
            obspy.UTCDateTime(2006, 11, 22, 13))
        self.assertEqual(channel["end_date"], None)

    def test_reading_more_complex_resp_file(self):
        """
        Tests reading a slightly more complex RESP file. This one contains two
        (identical!!!) channels. That is pretty much the maximum complexity of
        RESP files.
        """
        filename = os.path.join(self.data_dir, "RESP.AF.DODT..BHE")
        channels = simple_resp_parser.get_inventory(filename)
        self.assertEqual(len(channels), 2)
        channel = channels[0]
        self.assertEqual(channel["network"], "AF")
        self.assertEqual(channel["station"], "DODT")
        self.assertEqual(channel["location"], "")
        self.assertEqual(channel["channel"], "BHE")
        self.assertEqual(channel["start_date"],
            obspy.UTCDateTime(2009, 8, 9, 0))
        self.assertEqual(channel["end_date"], None)
        # The two have the same information for some reason...
        channel = channels[1]
        self.assertEqual(channel["network"], "AF")
        self.assertEqual(channel["station"], "DODT")
        self.assertEqual(channel["location"], "")
        self.assertEqual(channel["channel"], "BHE")
        self.assertEqual(channel["start_date"],
            obspy.UTCDateTime(2009, 8, 9, 0))
        self.assertEqual(channel["end_date"], None)


def suite():
    return unittest.makeSuite(RESPFileParserTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
