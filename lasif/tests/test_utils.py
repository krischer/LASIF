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
from obspy import UTCDateTime
from obspy.xseed import Parser
import os
import unittest

from lasif import utils


class UtilTestCase(unittest.TestCase):
    """
    Some heterogeneous tests.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_channel_in_parser(self):
        """
        Tests if a given channel is part of a Parser object.
        """
        starttime = UTCDateTime(2007, 2, 12, 10, 30, 28, 197700)
        endtime = UTCDateTime(2007, 2, 12, 11, 35, 28, 197700)
        channel_id = "ES.ECAL..HHE"
        # An empty file should of course not contain much.
        parser_object = Parser(os.path.join(self.data_dir,
            "channelless_datalessSEED"))
        self.assertFalse(utils.channel_in_parser(parser_object, channel_id,
            starttime, endtime))
        # Now read a file that actually contains data.
        channel_id = "IU.PAB.00.BHE"
        starttime = UTCDateTime(1999, 2, 18, 10, 0)
        endtime = UTCDateTime(2009, 8, 13, 19, 0)
        parser_object = Parser(os.path.join(self.data_dir,
            "dataless.IU_PAB"))
        # This is an exact fit of the start and end times in this file.
        self.assertTrue(utils.channel_in_parser(parser_object, channel_id,
            starttime, endtime))
        # Now try some others that do not fit.
        self.assertFalse(utils.channel_in_parser(parser_object, channel_id,
            starttime - 1, endtime))
        self.assertFalse(utils.channel_in_parser(parser_object, channel_id,
            starttime, endtime + 1))
        self.assertFalse(utils.channel_in_parser(parser_object,
            channel_id + "x", starttime, endtime))
        self.assertFalse(utils.channel_in_parser(parser_object, channel_id,
            starttime - 200, starttime - 100))
        self.assertFalse(utils.channel_in_parser(parser_object, channel_id,
            endtime + 100, endtime + 200))
        # And some that do fit.
        self.assertTrue(utils.channel_in_parser(parser_object, channel_id,
            starttime, starttime + 10))
        self.assertTrue(utils.channel_in_parser(parser_object, channel_id,
            endtime - 100, endtime))


def suite():
    return unittest.makeSuite(UtilTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
