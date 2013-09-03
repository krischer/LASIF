#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the station cache.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import os
import shutil
import time
import tempfile
import unittest

from lasif.tools.station_cache import StationCache


class StationCacheTest(unittest.TestCase):
    """
    One attempt at testing the station cache. Basically just one large test as
    things need to happen in order. Better then nothing.
    """
    @classmethod
    def setUpClass(cls):
        """
        Create the test case directories and appropriate files.
        """
        # Most generic way to get the actual data directory.
        cls.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

        # Create a temporary directory.
        cls.directory = tempfile.mkdtemp()

        cls.cache_file = os.path.join(cls.directory, "cache.sqlite")
        cls.seed_directory = os.path.join(cls.directory, "SEED")
        cls.resp_directory = os.path.join(cls.directory, "RESP")
        os.makedirs(cls.seed_directory)
        os.makedirs(cls.resp_directory)

        # Copy the SEED file. This files contains exactly one channel,
        # IU.PAB.00.BHE.
        shutil.copy(os.path.join(cls.data_dir, "dataless.IU_PAB"),
                    os.path.join(cls.seed_directory, "dataless.IU_PAB"))

    def test_station_cache(self):
        """
        Single test case checking the basic workflow.
        """
        # Init the station cache.
        station_cache = StationCache(self.cache_file, self.seed_directory,
                                     self.resp_directory)
        # Get the list of available channels.
        channels = station_cache.get_channels()
        # Check that the correct station is in there. Right now the folder only
        # contains one station.
        self.assertEqual(len(channels), 1)
        self.assertEqual(len(channels["IU.PAB.00.BHE"]), 1)

        del station_cache

        # This SEED file contains three more channels.
        seed_file = os.path.join(self.seed_directory, "dataless.BW_FURT")
        # Copy one more SEED file and check if the changes are reflected.
        shutil.copy(os.path.join(self.data_dir, "dataless.BW_FURT"),
                    seed_file)
        # Init the station cache once more.
        station_cache = StationCache(self.cache_file, self.seed_directory,
                                     self.resp_directory)
        # Get the list of available channels. It should not contain 4 channels.
        channels = station_cache.get_channels()
        self.assertEqual(len(channels), 4)
        self.assertTrue("BW.FURT..EHE" in channels)
        self.assertTrue("BW.FURT..EHN" in channels)
        self.assertTrue("BW.FURT..EHZ" in channels)
        # Now attempt to only retrieve the stations. It should have 2 stations.
        # One from each SEED file.
        stations = station_cache.get_stations()
        self.assertEqual(len(stations), 2)

        del station_cache

        # Delete the file, and check if everything else is removed as well. It
        # should not only contain one channel.
        os.remove(seed_file)
        station_cache = StationCache(self.cache_file, self.seed_directory,
                                     self.resp_directory)
        # Get the list of available channels.
        channels = station_cache.get_channels()
        # Check that the correct station is in there.
        self.assertEqual(len(channels), 1)
        self.assertEqual(len(channels["IU.PAB.00.BHE"]), 1)

        # Add the file once again...
        del station_cache
        shutil.copy(os.path.join(self.data_dir, "dataless.BW_FURT"),
                    seed_file)
        station_cache = StationCache(self.cache_file, self.seed_directory,
                                     self.resp_directory)
        # It should now again contain 4 channels.
        channels = station_cache.get_channels()
        self.assertEqual(len(channels), 4)
        del station_cache

        # The tolerance in last modified time is 0.2.
        time.sleep(0.2)
        # Now replace the file with an empty SEED file and assure that all
        # associated channels have been removed.
        shutil.copy(os.path.join(self.data_dir, "channelless_datalessSEED"),
                    seed_file)
        # Init the station cache once more.
        station_cache = StationCache(self.cache_file, self.seed_directory,
                                     self.resp_directory)
        # Get the list of available channels.
        channels = station_cache.get_channels()
        # Check that the correct station is in there.
        self.assertEqual(len(channels), 1)
        self.assertEqual(len(channels["IU.PAB.00.BHE"]), 1)

        del station_cache

        # Now copy some RESP files.
        resp_file = os.path.join(self.resp_directory, "RESP.G.FDF.00.BHE")
        shutil.copy(os.path.join(self.data_dir,
                                 os.path.basename(resp_file)), resp_file)
        # Init the station cache once more.
        station_cache = StationCache(self.cache_file, self.seed_directory,
                                     self.resp_directory)
        # Get the list of available channels.
        channels = station_cache.get_channels()
        # Check that the correct station is in there.
        self.assertEqual(len(channels), 2)
        self.assertTrue("IU.PAB.00.BHE" in channels)
        self.assertTrue("G.FDF.00.BHE" in channels)
        # Also get the stations once again.
        stations = station_cache.get_stations()
        self.assertEqual(len(stations), 2)

        del station_cache

        # Add some more RESP files.
        shutil.copy(
            os.path.join(self.data_dir, "RESP.AF.DODT..BHE"),
            os.path.join(self.resp_directory, "RESP.AF.DODT..BHE"))
        shutil.copy(
            os.path.join(self.data_dir, "RESP.G.FDF.00.BHN"),
            os.path.join(self.resp_directory, "RESP.G.FDF.00.BHN"))
        shutil.copy(
            os.path.join(self.data_dir, "RESP.G.FDF.00.BHZ"),
            os.path.join(self.resp_directory, "RESP.G.FDF.00.BHZ"))
        # Init the station cache once more.
        station_cache = StationCache(self.cache_file, self.seed_directory,
                                     self.resp_directory)
        # Get the list of available channels.
        channels = station_cache.get_channels()
        # Check that the correct station is in there.
        self.assertEqual(len(channels), 5)
        self.assertTrue("IU.PAB.00.BHE" in channels)
        self.assertTrue("G.FDF.00.BHE" in channels)
        self.assertTrue("G.FDF.00.BHN" in channels)
        self.assertTrue("G.FDF.00.BHZ" in channels)
        self.assertTrue("AF.DODT..BHE" in channels)
        # The duplicates in the one RESP files should not show up.
        self.assertEqual(len(channels["AF.DODT..BHE"]), 1)
        # Also get the stations once again.
        stations = station_cache.get_stations()
        self.assertEqual(len(stations), 3)

        # Check the get_values() method.
        all_values = station_cache.get_values()
        self.assertEqual(len(all_values), 5)

        # Test the retrieval of only a single record by its filename.
        single_value = station_cache.get_details(all_values[0]["filename"])[0]
        self.assertEqual(single_value, all_values[0])

    @classmethod
    def tearDownClass(cls):
        """
        Remove the temporary directory and all contents.
        """
        shutil.rmtree(cls.directory)


def suite():
    return unittest.makeSuite(StationCacheTest, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
