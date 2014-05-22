import inspect
import obspy
import os
import pytest

from lasif import LASIFNotFoundError
from lasif.components.stations import StationsComponent
from lasif.components.communicator import Communicator

# Test data directory
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))), "data")


def _create_stations_object(cache_directory):
    """
    Helper function returning an initialized stations component.
    """
    comm = Communicator()
    stations_component = StationsComponent(
        stationxml_folder=os.path.join(data_dir, "station_files",
                                       "stationxml"),
        seed_folder=os.path.join(data_dir, "station_files", "seed"),
        resp_folder=os.path.join(data_dir, "station_files", "resp"),
        cache_folder=cache_directory,
        communicator=comm,
        component_name="stations")
    return stations_component, comm


def test_station_cache_update(tmpdir):
    tmpdir = str(tmpdir)
    comm = _create_stations_object(tmpdir)[1]
    assert len(os.listdir(tmpdir)) == 0
    comm.stations.force_cache_update()
    assert os.listdir(tmpdir) == ["station_cache.sqlite"]

    # Make sure all files are in there.
    filenames = list(set([i["filename"]
                          for i in comm.stations.get_all_channels()]))
    assert sorted(os.path.basename(i) for i in filenames) == \
           sorted(
               ["RESP.AF.DODT..BHE", "RESP.G.FDF.00.BHE", "RESP.G.FDF.00.BHN",
                "RESP.G.FDF.00.BHZ", "dataless.BW_FURT", "dataless.IU_PAB",
                "IRIS_single_channel_with_response.xml"])

    # Deleting the cache file and initializing a new stations object will
    # create the station cache if it is accessed.
    os.remove(os.path.join(tmpdir, "station_cache.sqlite"))
    assert len(os.listdir(tmpdir)) == 0
    comm = _create_stations_object(tmpdir)[1]
    filenames = list(set([i["filename"]
                          for i in comm.stations.get_all_channels()]))
    assert sorted(os.path.basename(i) for i in filenames) == \
           sorted(
               ["RESP.AF.DODT..BHE", "RESP.G.FDF.00.BHE", "RESP.G.FDF.00.BHN",
                "RESP.G.FDF.00.BHZ", "dataless.BW_FURT", "dataless.IU_PAB",
                "IRIS_single_channel_with_response.xml"])
    assert os.listdir(tmpdir) == ["station_cache.sqlite"]


def test_has_channel(tmpdir):
    comm = _create_stations_object(str(tmpdir))[1]
    all_channels = comm.stations.get_all_channels()
    for channel in all_channels:
        # Works for timestamps and UTCDateTime objects.
        assert comm.stations.has_channel(channel["channel_id"],
                                         channel["start_date"])
        assert comm.stations.has_channel(
            channel["channel_id"], obspy.UTCDateTime(channel["start_date"]))
        # Slightly after the starttime should still work.
        assert comm.stations.has_channel(channel["channel_id"],
                                         channel["start_date"] + 3600)
        assert comm.stations.has_channel(
            channel["channel_id"],
            obspy.UTCDateTime(channel["start_date"] + 3600))
        # Slightly before not.
        assert not comm.stations.has_channel(channel["channel_id"],
                                             channel["start_date"] - 3600)
        assert not comm.stations.has_channel(
            channel["channel_id"],
            obspy.UTCDateTime(channel["start_date"] - 3600))

        # For those that have an endtime, do the same.
        if channel["end_date"]:
            assert comm.stations.has_channel(channel["channel_id"],
                                             channel["end_date"])
            assert comm.stations.has_channel(
                channel["channel_id"],
                obspy.UTCDateTime(channel["end_date"]))
            # Slightly before.
            assert comm.stations.has_channel(channel["channel_id"],
                                             channel["end_date"] - 3600)
            assert comm.stations.has_channel(
                channel["channel_id"],
                obspy.UTCDateTime(channel["end_date"] - 3600))
            # But not slightly after.
            assert not comm.stations.has_channel(channel["channel_id"],
                                                 channel["end_date"] + 3600)
            assert not comm.stations.has_channel(
                channel["channel_id"],
                obspy.UTCDateTime(channel["end_date"] + 3600))
        else:
            # For those that do not have an endtime, a time very far in the
            # future should work just fine.
            assert comm.stations.has_channel(channel["channel_id"],
                                             obspy.UTCDateTime(2030, 1, 1))

def test_get_station_filename(tmpdir):
    comm = _create_stations_object(str(tmpdir))[1]
    all_channels = comm.stations.get_all_channels()
    for channel in all_channels:
        # Should work for timestamps and UTCDateTime objects.
        assert channel["filename"] == comm.stations.get_station_filename(
            channel["channel_id"], channel["start_date"] + 3600)
        assert channel["filename"] == comm.stations.get_station_filename(
            channel["channel_id"],
            obspy.UTCDateTime(channel["start_date"] + 3600))
        with pytest.raises(LASIFNotFoundError):
            comm.stations.get_station_filename(
                channel["channel_id"], channel["start_date"] - 3600)


def test_get_coordinates_for_stations(tmpdir):
    comm = _create_stations_object(str(tmpdir))[1]
    all_channels = comm.stations.get_all_channels()
    for channel in all_channels:
        net, sta = channel["channel_id"].split(".")[:2]
        # Resp files have no coordinates.
        if "RESP." in channel["filename"]:
            with pytest.raises(LASIFNotFoundError):
                comm.stations.get_coordinates(net, sta)
        # Others do.
        else:
            coordinates = comm.stations.get_coordinates(net, sta)
            assert coordinates == {
                "latitude": channel["latitude"],
                "longitude": channel["longitude"],
                "elevation_in_m": channel["elevation_in_m"],
                "local_depth_in_m": channel["local_depth_in_m"]}
    # Not available station.
    with pytest.raises(LASIFNotFoundError):
        comm.stations.get_coordinates("AA", "BB")


def test_get_details_for_filename(tmpdir):
    comm = _create_stations_object(str(tmpdir))[1]
    all_channels = comm.stations.get_all_channels()
    for channel in all_channels:
        assert channel in \
           comm.stations.get_details_for_filename(channel["filename"])


def test_all_coordinates_at_time(tmpdir):
    comm = _create_stations_object(str(tmpdir))[1]
    all_channels = comm.stations.get_all_channels()

    # There is only one stations that start that early.
    coords = comm.stations.get_all_coordinates_at_time(920000000)
    assert coords == \
    {'IU.PAB.00.BHE':
         {'latitude': 39.5446, 'elevation_in_m': 950.0,
          'local_depth_in_m': 0.0, 'longitude': -4.349899}}
    # Also works with a UTCDateTime object.
    coords = comm.stations.get_all_coordinates_at_time(
        obspy.UTCDateTime(920000000))
    assert coords == \
           {'IU.PAB.00.BHE':
                {'latitude': 39.5446, 'elevation_in_m': 950.0,
                 'local_depth_in_m': 0.0, 'longitude': -4.349899}}

    # Most channels have no set endtime or an endtime very far in the
    # future.
    channels = comm.stations.get_all_coordinates_at_time(
        obspy.UTCDateTime(2030, 1, 1))
    assert sorted(channels.keys()) == sorted(
        ["G.FDF.00.BHZ", "G.FDF.00.BHN", "G.FDF.00.BHE", "AF.DODT..BHE",
         "BW.FURT..EHE", "BW.FURT..EHN", "BW.FURT..EHZ", "IU.ANMO.10.BHZ"])
