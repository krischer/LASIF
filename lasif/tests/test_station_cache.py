#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the station cache.

One attempt at testing the station cache. Basically just one large test as
things need to happen in order. Better then nothing.

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

from lasif.tools.station_cache import StationCache


def test_station_cache(tmpdir):
    """
    Single test case checking the basic workflow.
    """
    # Most generic way to get the actual data directory.
    data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe()))), "data")

    # Create a temporary directory.
    directory = tempfile.mkdtemp()

    cache_file = os.path.join(directory, "cache.sqlite")
    seed_directory = os.path.join(directory, "SEED")
    resp_directory = os.path.join(directory, "RESP")
    stationxml_directory = os.path.join(directory, "StationXML")
    os.makedirs(seed_directory)
    os.makedirs(resp_directory)

    # Copy the SEED file. This files contains exactly one channel,
    # IU.PAB.00.BHE.
    shutil.copy(os.path.join(data_dir, "dataless.IU_PAB"),
                os.path.join(seed_directory, "dataless.IU_PAB"))

    # Init the station cache.
    station_cache = StationCache(cache_file, seed_directory, resp_directory,
                                 stationxml_directory)
    # Get the list of available channels.
    channels = station_cache.get_channels()
    # Check that the correct station is in there. Right now the folder only
    # contains one station.
    assert len(channels) == 1
    assert len(channels["IU.PAB.00.BHE"]) == 1

    del station_cache

    # This SEED file contains three more channels.
    seed_file = os.path.join(seed_directory, "dataless.BW_FURT")
    # Copy one more SEED file and check if the changes are reflected.
    shutil.copy(os.path.join(data_dir, "dataless.BW_FURT"), seed_file)
    # Init the station cache once more.
    station_cache = StationCache(cache_file, seed_directory, resp_directory,
                                 stationxml_directory)
    # Get the list of available channels. It should not contain 4 channels.
    channels = station_cache.get_channels()
    assert len(channels) == 4
    assert "BW.FURT..EHE" in channels
    assert "BW.FURT..EHN" in channels
    assert "BW.FURT..EHZ" in channels
    # Now attempt to only retrieve the stations. It should have 2 stations.
    # One from each SEED file.
    stations = station_cache.get_stations()
    assert len(stations) == 2

    del station_cache

    # Delete the file, and check if everything else is removed as well. It
    # should not only contain one channel.
    os.remove(seed_file)
    station_cache = StationCache(cache_file, seed_directory, resp_directory,
                                 stationxml_directory)
    # Get the list of available channels.
    channels = station_cache.get_channels()
    # Check that the correct station is in there.
    assert len(channels) == 1
    assert len(channels["IU.PAB.00.BHE"]) == 1

    # Add the file once again...
    del station_cache
    shutil.copy(os.path.join(data_dir, "dataless.BW_FURT"), seed_file)
    station_cache = StationCache(cache_file, seed_directory, resp_directory,
                                 stationxml_directory)
    # It should now again contain 4 channels.
    channels = station_cache.get_channels()
    assert len(channels) == 4
    del station_cache

    # The tolerance in last modified time is 0.2. Set it much higher as
    # machines occasionaly hick up for some reason.
    time.sleep(1.5)
    # Now replace the file with an empty SEED file and assure that all
    # associated channels have been removed.
    shutil.copy(os.path.join(data_dir, "channelless_datalessSEED"), seed_file)
    # Init the station cache once more.
    station_cache = StationCache(cache_file, seed_directory, resp_directory,
                                 stationxml_directory)
    # Get the list of available channels.
    channels = station_cache.get_channels()
    # Check that the correct station is in there.
    assert len(channels) == 1
    assert len(channels["IU.PAB.00.BHE"]) == 1

    del station_cache

    # Now copy some RESP files.
    resp_file = os.path.join(resp_directory, "RESP.G.FDF.00.BHE")
    shutil.copy(os.path.join(data_dir, os.path.basename(resp_file)), resp_file)
    # Init the station cache once more.
    station_cache = StationCache(cache_file, seed_directory, resp_directory,
                                 stationxml_directory)
    # Get the list of available channels.
    channels = station_cache.get_channels()
    # Check that the correct station is in there.
    assert len(channels) == 2
    assert "IU.PAB.00.BHE" in channels
    assert "G.FDF.00.BHE" in channels
    # Also get the stations once again.
    stations = station_cache.get_stations()
    assert len(stations) == 2

    del station_cache

    # Add some more RESP files.
    shutil.copy(os.path.join(data_dir, "RESP.AF.DODT..BHE"),
                os.path.join(resp_directory, "RESP.AF.DODT..BHE"))
    shutil.copy(os.path.join(data_dir, "RESP.G.FDF.00.BHN"),
                os.path.join(resp_directory, "RESP.G.FDF.00.BHN"))
    shutil.copy(os.path.join(data_dir, "RESP.G.FDF.00.BHZ"),
                os.path.join(resp_directory, "RESP.G.FDF.00.BHZ"))
    # Init the station cache once more.
    station_cache = StationCache(cache_file, seed_directory, resp_directory,
                                 stationxml_directory)
    # Get the list of available channels.
    channels = station_cache.get_channels()
    # Check that the correct station is in there.
    assert len(channels) == 5
    assert "IU.PAB.00.BHE" in channels
    assert "G.FDF.00.BHE" in channels
    assert "G.FDF.00.BHN" in channels
    assert "G.FDF.00.BHZ" in channels
    assert "AF.DODT..BHE" in channels
    # The duplicates in the one RESP files should not show up.
    assert len(channels["AF.DODT..BHE"]) == 1
    # Also get the stations once again.
    stations = station_cache.get_stations()
    assert len(stations) == 3

    # Check the get_values() method.
    all_values = station_cache.get_values()
    assert len(all_values) == 5

    # Test the retrieval of only a single record by its filename.
    single_value = station_cache.get_details(all_values[0]["filename"])[0]
    assert single_value == all_values[0]

    coordinates = \
        station_cache.get_coordinates_for_station(network="IU", station="PAB")
    assert coordinates == {"latitude": 39.5446, "longitude": -4.349899,
                           "elevation_in_m": 950.0, "local_depth_in_m": 0.0}
