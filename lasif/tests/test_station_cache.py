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
import itertools
import os
import shutil
import time

import obspy

from lasif import LASIFWarning
from lasif.tools.cache_helpers.station_cache import StationCache


def test_station_cache(tmpdir):
    """
    Single test case checking the basic workflow.
    """
    # Most generic way to get the actual data directory.
    data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe()))), "data", "station_files")

    # Create a temporary directory.
    directory = str(tmpdir)

    cache_file = os.path.join(directory, "cache.sqlite")
    seed_directory = os.path.join(directory, "SEED")
    resp_directory = os.path.join(directory, "RESP")
    stationxml_directory = os.path.join(directory, "StationXML")
    os.makedirs(seed_directory)
    os.makedirs(resp_directory)

    # Copy the SEED file. This files contains exactly one channel,
    # IU.PAB.00.BHE.
    shutil.copy(os.path.join(data_dir, "seed", "dataless.IU_PAB"),
                os.path.join(seed_directory, "dataless.IU_PAB"))

    # Init the station cache.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
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
    shutil.copy(os.path.join(data_dir, "seed", "dataless.BW_FURT"), seed_file)
    # Init the station cache once more.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
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

    # Test the file_count, index_count, and total_size properties.
    assert station_cache.file_count == 2
    assert station_cache.index_count == 4
    assert station_cache.total_size == 12288 + 28672

    del station_cache

    # Delete the file, and check if everything else is removed as well. It
    # should not only contain one channel.
    os.remove(seed_file)
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
    # Get the list of available channels.
    channels = station_cache.get_channels()
    # Check that the correct station is in there.
    assert len(channels) == 1
    assert len(channels["IU.PAB.00.BHE"]) == 1

    # Add the file once again...
    del station_cache
    shutil.copy(os.path.join(data_dir, "seed", "dataless.BW_FURT"), seed_file)
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
    # It should now again contain 4 channels.
    channels = station_cache.get_channels()
    assert len(channels) == 4
    del station_cache

    # The tolerance in last modified time is 0.2. Set it much higher as
    # machines occasionaly hick up for some reason.
    time.sleep(1.5)
    # Now replace the file with an empty SEED file and assure that all
    # associated channels have been removed.
    shutil.copy(os.path.join(data_dir, "seed", "channelless_datalessSEED"),
                seed_file)
    # Init the station cache once more.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
    # Get the list of available channels.
    channels = station_cache.get_channels()
    # Check that the correct station is in there.
    assert len(channels) == 1
    assert len(channels["IU.PAB.00.BHE"]) == 1

    del station_cache

    # Now copy some RESP files.
    resp_file = os.path.join(resp_directory, "RESP.G.FDF.00.BHE")
    shutil.copy(os.path.join(data_dir, "resp", os.path.basename(resp_file)),
                resp_file)
    # Init the station cache once more.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
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
    shutil.copy(os.path.join(data_dir, "resp", "RESP.AF.DODT..BHE"),
                os.path.join(resp_directory, "RESP.AF.DODT..BHE"))
    shutil.copy(os.path.join(data_dir, "resp", "RESP.G.FDF.00.BHN"),
                os.path.join(resp_directory, "RESP.G.FDF.00.BHN"))
    shutil.copy(os.path.join(data_dir, "resp", "RESP.G.FDF.00.BHZ"),
                os.path.join(resp_directory, "RESP.G.FDF.00.BHZ"))
    # Init the station cache once more.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
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

    single_value = station_cache.get_details(all_values[0]["filename"])[0]
    assert single_value == all_values[0]


def test_station_xml(tmpdir):
    """
    Tests the StationXML support.
    """
    # Most generic way to get the actual data directory.
    data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe()))), "data", "station_files")

    # Create a temporary directory.
    directory = str(tmpdir)

    cache_file = os.path.join(directory, "cache.sqlite")
    seed_directory = os.path.join(directory, "SEED")
    resp_directory = os.path.join(directory, "RESP")
    stationxml_directory = os.path.join(directory, "StationXML")
    os.makedirs(seed_directory)
    os.makedirs(resp_directory)
    os.makedirs(stationxml_directory)

    # Copy the StationXML file.
    shutil.copy(os.path.join(data_dir, "stationxml",
                             "IRIS_single_channel_with_response.xml"),
                os.path.join(stationxml_directory,
                             "IRIS_single_channel_with_response.xml"))

    # Init station cache.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
    stations = station_cache.get_stations()
    channels = station_cache.get_channels()
    filename = station_cache.get_station_filename(
        "IU.ANMO.10.BHZ", obspy.UTCDateTime(2013, 1, 1))

    assert len(stations) == 1
    assert stations == {"IU.ANMO": {"latitude": 34.945913,
                                    "longitude": -106.457122}}
    assert len(channels) == 1
    assert channels == {"IU.ANMO.10.BHZ": [
        {"local_depth_in_m": 57.0, "endtime_timestamp": 19880899199,
         "elevation_in_m": 1759.0, "startime_timestamp": 1331626200,
         "longitude": -106.457122, "latitude": 34.945913}]}
    assert filename == os.path.join(
        stationxml_directory, "IRIS_single_channel_with_response.xml")


def test_exception_handling(tmpdir, recwarn):
    """
    Tests exception handling.

    For each file that somehow failed to index, it should raise a warning
    while still indexing the rest.
    """
    # Most generic way to get the actual data directory.
    data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe()))), "data", "station_files")

    # Create a temporary directory.
    directory = str(tmpdir)

    cache_file = os.path.join(directory, "cache.sqlite")
    seed_directory = os.path.join(directory, "SEED")
    resp_directory = os.path.join(directory, "RESP")
    stationxml_directory = os.path.join(directory, "StationXML")

    os.makedirs(seed_directory)
    os.makedirs(resp_directory)
    os.makedirs(stationxml_directory)

    # Copy a StationXML file to a RESP directory which naturally results in
    # an error which results in a triggered warnings.
    shutil.copy(os.path.join(data_dir, "stationxml",
                             "IRIS_single_channel_with_response.xml"),
                os.path.join(resp_directory,
                             "RESP.file"))

    recwarn.clear()
    StationCache(cache_file, directory, seed_directory, resp_directory,
                 stationxml_directory, read_only=False)
    w = recwarn.pop(LASIFWarning)
    assert "Failed to index" in str(w.message)
    assert "Not a valid RESP file?" in str(w.message)

    # Clear the directories and create a new station cache instance using some
    # correct and some incorrect files.
    os.remove(cache_file)
    os.remove(os.path.join(resp_directory, "RESP.file"))

    # A valid SEED file.
    shutil.copy(os.path.join(data_dir, "seed", "dataless.IU_PAB"),
                os.path.join(seed_directory, "dataless.IU_PAB"))
    # A valid RESP file.
    shutil.copy(os.path.join(data_dir, "resp", "RESP.G.FDF.00.BHE"),
                os.path.join(resp_directory, "RESP.G.FDF.00.BHE"))
    # A valid StationXML file.
    shutil.copy(os.path.join(data_dir, "stationxml",
                             "IRIS_single_channel_with_response.xml"),
                os.path.join(stationxml_directory,
                             "IRIS_single_channel_with_response.xml"))

    # Now copy files of the wrong types to the folders.
    shutil.copy(os.path.join(data_dir, "seed", "dataless.IU_PAB"),
                os.path.join(resp_directory, "RESP.iu_pab"))
    shutil.copy(os.path.join(data_dir, "seed", "dataless.IU_PAB"),
                os.path.join(stationxml_directory, "IU_PAB.xml"))
    shutil.copy(os.path.join(data_dir, "resp", "RESP.G.FDF.00.BHE"),
                os.path.join(seed_directory, "dataless.G_FDF"))

    # Also create two StationXML files, one missing the response and one
    # also missing the channel.
    inv = obspy.read_inventory(os.path.join(
        data_dir, "stationxml", "IRIS_single_channel_with_response.xml"))
    inv[0][0][0].response = None
    inv[0].code = "AA"
    inv.write(os.path.join(stationxml_directory, "dummy_1.xml"),
              format="stationxml")
    inv[0][0].channels = []
    inv[0].code = "BB"
    inv.write(os.path.join(stationxml_directory, "dummy_2.xml"),
              format="stationxml")

    # Clear all warnings and init a new StationCache.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
    # It should still have indexed all three files and thus three channels
    # should be present.
    assert len(station_cache.get_channels()) == 3
    assert sorted(["G.FDF.00.BHE", "IU.ANMO.10.BHZ", "IU.PAB.00.BHE"]) == \
        sorted(station_cache.get_channels().keys())
    # Same for the stations.
    assert len(station_cache.get_stations()) == 3
    assert sorted(["G.FDF", "IU.ANMO", "IU.PAB"]) == \
        sorted(station_cache.get_stations().keys())

    # Naturally only three files should have been indexed.
    filenames = list(itertools.chain(*station_cache.files.values()))
    assert len(filenames) == 3

    # 5 warnings should have been raised.
    assert len(recwarn.list) == 5
    messages = sorted([str(_w.message).split(":")[-1].strip()
                       for _w in recwarn.list])
    test_strings = sorted([
        "Not a valid SEED file?",
        "Not a valid RESP file?",
        "Not a valid StationXML file?",
        "Channel AA.ANMO.10.BHZ has no response.",
        "File has no channels."
    ])
    assert messages == test_strings


def test_station_cache_readonly_mode(tmpdir):
    """
    Tests the readonly mode of the station cache.

    There appears to be no simple way to check if the database is actually
    opened in read-only mode without using a different database wrapper. So
    this will have to do for now.
    """
    # Most generic way to get the actual data directory.
    data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe()))), "data", "station_files")

    # Create a temporary directory.
    directory = str(tmpdir)

    cache_file = os.path.join(directory, "cache.sqlite")
    seed_directory = os.path.join(directory, "SEED")
    resp_directory = os.path.join(directory, "RESP")
    stationxml_directory = os.path.join(directory, "StationXML")
    os.makedirs(resp_directory)

    # Add some more RESP files.
    shutil.copy(os.path.join(data_dir, "resp", "RESP.AF.DODT..BHE"),
                os.path.join(resp_directory, "RESP.AF.DODT..BHE"))
    shutil.copy(os.path.join(data_dir, "resp", "RESP.G.FDF.00.BHN"),
                os.path.join(resp_directory, "RESP.G.FDF.00.BHN"))
    shutil.copy(os.path.join(data_dir, "resp", "RESP.G.FDF.00.BHZ"),
                os.path.join(resp_directory, "RESP.G.FDF.00.BHZ"))
    # Init the station cache once more.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=False)
    original = station_cache.get_values()

    # Now open the same database in read_only mode.
    station_cache = StationCache(cache_file, directory, seed_directory,
                                 resp_directory, stationxml_directory,
                                 read_only=True)
    new = station_cache.get_values()
    assert original == new
