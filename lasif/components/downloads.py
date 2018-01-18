#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import os

from .component import Component


class DownloadsComponent(Component):
    """
    Component dealing with the station and data downloading.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def download_data(self, event, providers=None):
        """
        """
        event = self.comm.events.get(event)
        from obspy.clients.fdsn.mass_downloader import MassDownloader, \
            Restrictions

        proj = self.comm.project

        domain = self._get_exodus_domain(proj.domain)
        event_time = event["origin_time"]
        ds = proj.config["download_settings"]
        starttime = event_time - ds["seconds_before_event"]
        endtime = event_time + ds["seconds_after_event"]
        event_name = event["event_name"]

        restrictions = Restrictions(
            starttime=starttime,
            endtime=endtime,
            # Go back 1 day.
            station_starttime=starttime - 86400 * 1,
            # Advance 1 day.
            station_endtime=endtime + 86400 * 1,
            network=None, station=None, location=None, channel=None,
            minimum_interstation_distance_in_m=ds[
                "interstation_distance_in_meters"],
            reject_channels_with_gaps=True,
            minimum_length=0.95,
            location_priorities=ds["location_priorities"],
            channel_priorities=ds["channel_priorities"])

        filename = proj.paths["eq_data"] / (event["event_name"] + ".h5")

        import pyasdf
        asdf_ds = pyasdf.ASDFDataSet(filename, compression="gzip-3")

        stationxml_storage_path = proj.paths["eq_data"] / \
            f"tmp_station_xml_storage_{event_name}"
        stationxml_storage = self._get_stationxml_storage_fct(
            asdf_ds, starttime, endtime, stationxml_storage_path)
        mseed_storage_path = proj.paths["eq_data"] / \
            f"tmp_mseed_storage_{event_name}"
        mseed_storage = self._get_mseed_storage_fct(asdf_ds, starttime,
                                                    endtime,
                                                    mseed_storage_path)

        # Also log to file for reasons of provenance and debugging.
        logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
        fh = logging.FileHandler(
            self.comm.project.get_log_file("DOWNLOADS", event_name))
        fh.setLevel(logging.INFO)
        FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(FORMAT)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        dlh = MassDownloader(providers=providers)
        dlh.download(domain=domain, restrictions=restrictions,
                     mseed_storage=mseed_storage,
                     stationxml_storage=stationxml_storage)

        import glob
        files = glob.glob(str(mseed_storage_path / "*.mseed"))
        for _i, filename in enumerate(files):
            print("Adding raw_recording %i of %i ..." % (_i + 1, len(files)))

            try:
                asdf_ds.add_waveforms(filename, tag="raw_recording",
                                      event_id=asdf_ds.events[0])
            except Exception as e:
                print(e)

        files = glob.glob(str(stationxml_storage_path / '*.xml'))

        for _i, filename in enumerate(files):
            print("Adding stationxml %i of %i ..." % (_i + 1, len(files)))
            try:
                asdf_ds.add_stationxml(filename)
            except Exception as e:
                print(e)

        # remove stations with missing information
        for station in asdf_ds.waveforms.list():
            station_inventory = asdf_ds.waveforms[station].list()
            if "StationXML" not in station_inventory or \
                    len(station_inventory) < 2:
                del asdf_ds.waveforms[station]
                msg = f"Removed station {station} due to missing " \
                      f"StationXMl or waveform information."
                print(msg)
                continue

            # if coordinates are specified at the channel and are different
            # skip and remove, perhaps add a tolerance here in the future.
            else:
                obspy_station = asdf_ds.waveforms[station].StationXML[0][0]
                if not obspy_station.channels:
                    continue
                else:
                    coords = set(
                        (_i.latitude, _i.longitude, _i.depth) for _i in
                        obspy_station.channels)
                    if len(coords) != 1:
                        del asdf_ds.waveforms[station]
                        msg = f"Removed station {station} due to " \
                              f"inconsistent channel coordinates."
                        print(msg)

        for station in asdf_ds.waveforms.list():
            obspy_inv = asdf_ds.waveforms[station].StationXML
            lat = obspy_inv.get_coordinates(
                obspy_inv.get_contents()['channels'][0])['latitude']
            if lat > 40.0:
                del asdf_ds.waveforms[station]
                print(f"deleted station {station}")

        # clean up temporary download directories
        import shutil
        if os.path.exists(stationxml_storage_path):
            shutil.rmtree(stationxml_storage_path)
        if os.path.exists(mseed_storage_path):
            shutil.rmtree(mseed_storage_path)

    def _get_mseed_storage_fct(self, ds, starttime, endtime, storage_path):

        def get_mseed_storage(network, station, location, channel, starttime,
                              endtime):
            # Returning True means that neither the data nor the
            # StationXML file will be downloaded.
            net_sta = f"{network}.{station}"
            if net_sta in ds.waveforms.list() and \
                "raw_recording" in ds.waveforms[net_sta] and \
                ds.waveforms[net_sta].raw_recording.select(
                    network=network, station=station, location=location,
                    channel=channel):
                return True
            return str(storage_path /
                       f"{network}.{station}.{location}.{channel}.mseed")

        return get_mseed_storage

    def _get_stationxml_storage_fct(self, ds, starttime, endtime,
                                    storage_path):
        if not os.path.isdir(storage_path):
            os.mkdir(storage_path)

        def stationxml_storage(network, station, channels, startime, endtime):
            missing_channels = []
            available_channels = []
            for loc_code, cha_code in channels:
                net_sta = f"{network}.{station}"

                if net_sta in ds.waveforms.list() and "StationXML" in \
                    ds.waveforms[net_sta].list() and \
                        ds.waveforms[net_sta].StationXML.select(
                            network=network,
                            station=station, channel=cha_code):
                    available_channels.append((loc_code, cha_code))
                else:
                    missing_channels.append((loc_code, cha_code))

            _i = 0
            while True:
                path = os.path.join(storage_path, "%s.%s%s.xml" % (
                    network, station, _i if _i >= 1 else ""))
                if os.path.exists(path):
                    _i += 1
                    continue
                break

            return {
                "available_channels": available_channels,
                "missing_channels": missing_channels,
                "filename": path
            }

        return stationxml_storage

    def _get_exodus_domain(self, domain):
        from obspy.clients.fdsn.mass_downloader import Domain

        class ExodusDomain(Domain):

            def get_query_parameters(self):
                return {}

            def is_in_domain(self, latitude, longitude):
                return domain.point_in_domain(latitude=latitude,
                                              longitude=longitude)

        return ExodusDomain()
