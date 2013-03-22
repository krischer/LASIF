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

from fwiw import utils


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
        parser_object = Parser(os.path.join(self.data_dir,
            "channelless_datalessSEED"))
        self.assertFalse(utils.channel_in_parser(parser_object, channel_id,
            starttime, endtime))


def suite():
    return unittest.makeSuite(UtilTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
