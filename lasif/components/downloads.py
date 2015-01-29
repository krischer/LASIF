#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import logging
import numpy as np
import os

from lasif import rotations
import lasif.domain
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
        try:
            from obspy.fdsn.download_helpers import DownloadHelper, \
                GlobalDomain, Restrictions
        except ImportError:
            raise ImportError("Currently requires the "
                              "krischer/download_helpers branch of ObsPy. "
                              "Should soon be either merged into ObsPy or"
                              "outsourced in a separate project.")

        proj = self.comm.project

        if isinstance(proj.domain, lasif.domain.GlobalDomain):
            domain = GlobalDomain()
        else:
            domain = self._get_spherical_section_domain(proj.domain)

        event_time = event["origin_time"]
        ds = proj.config["download_settings"]
        starttime = event_time - ds["seconds_before_event"]
        endtime = event_time + ds["seconds_after_event"]

        mseed_storage = os.path.join(proj.paths["data"], event["event_name"],
                                     "raw")
        restrictions = Restrictions(
            starttime=starttime,
            endtime=endtime,
            # Go back 10 years.
            station_starttime=starttime - 86400 * 365.25 * 10,
            # Advance 10 years.
            station_endtime=endtime + 86400 * 365.25 * 10,
            network=None, station=None, location=None, channel=None,
            minimum_interstation_distance_in_m=ds[
                "interstation_distance_in_m"],
            reject_channels_with_gaps=True,
            minimum_length=0.95,
            location_priorities=ds["location_priorities"],
            channel_priorities=ds["channel_priorities"])

        stationxml_storage = self._get_stationxml_storage_fct(starttime,
                                                              endtime)

        # Also log to file for reasons of provenance and debugging.
        logger = logging.getLogger("obspy-download-helper")
        fh = logging.FileHandler(
            self.comm.project.get_log_file("DOWNLOADS", event["event_name"]))
        fh.setLevel(logging.INFO)
        FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
        formatter = logging.Formatter(FORMAT)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        dlh = DownloadHelper(providers=providers)
        dlh.download(domain=domain, restrictions=restrictions,
                     mseed_storage=mseed_storage,
                     stationxml_storage=stationxml_storage)

    def _get_stationxml_storage_fct(self, starttime, endtime):
        # Get the stationxml storage function required by the download helpers.
        time_of_interest = starttime + 0.5 * (endtime - starttime)
        root_path = self.comm.project.paths["station_xml"]

        def stationxml_storage(network, station, channels, startime, endtime):
            missing_channels = []
            available_channels = []
            for loc_code, cha_code in channels:
                if not self.comm.stations.has_channel(
                    "%s.%s.%s.%s" % (network, station, loc_code,
                                     cha_code), time_of_interest):
                    missing_channels.append((loc_code, cha_code))
                else:
                    available_channels.append((loc_code, cha_code))
            _i = 0
            while True:
                path = os.path.join(root_path, "%s.%s%s.xml" % (
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

    def _get_spherical_section_domain(self, domain):
        from obspy.fdsn.download_helpers import Domain

        # Make copies to assure the closure binds correctly.
        d = copy.deepcopy(domain)
        min_lat = d.min_latitude
        max_lat = d.max_latitude
        min_lng = d.min_longitude
        max_lng = d.max_longitude
        rotation_angle = d.rotation_angle_in_degree
        rotation_axis = d.rotation_axis

        min_lat, max_lat, min_lng, max_lng = self._get_maximum_bounds(
            min_lat, max_lat, min_lng, max_lng,
            rotation_axis=rotation_axis,
            rotation_angle_in_degree=rotation_angle)

        class SphericalSectionDomain(Domain):
            def get_query_parameters(self):
                return {
                    "minlatitude": min_lat,
                    "maxlatitude": max_lat,
                    "minlongitude": min_lng,
                    "maxlongitude": max_lng
                }

            def is_in_domain(self, latitude, longitude):
                return d.point_in_domain(latitude=latitude,
                                         longitude=longitude)

        return SphericalSectionDomain()

    def _get_maximum_bounds(self, min_lat, max_lat, min_lng, max_lng,
                            rotation_axis, rotation_angle_in_degree):
        """
        Small helper function to get the domain bounds of a rotated spherical
        section.

        :param min_lat: Minimum Latitude of the unrotated section.
        :param max_lat: Maximum Latitude of the unrotated section.
        :param min_lng: Minimum Longitude of the unrotated section.
        :param max_lng: Maximum Longitude of the unrotated section.
        :param rotation_axis: Rotation axis as a list in the form of [x, y, z]
        :param rotation_angle_in_degree: Rotation angle in degree.
        """
        number_of_points_per_side = 50
        north_border = np.empty((number_of_points_per_side, 2))
        south_border = np.empty((number_of_points_per_side, 2))
        east_border = np.empty((number_of_points_per_side, 2))
        west_border = np.empty((number_of_points_per_side, 2))

        north_border[:, 0] = np.linspace(min_lng, max_lng,
                                         number_of_points_per_side)
        north_border[:, 1] = min_lat

        south_border[:, 0] = np.linspace(max_lng, min_lng,
                                         number_of_points_per_side)
        south_border[:, 1] = max_lat

        east_border[:, 0] = max_lng
        east_border[:, 1] = np.linspace(min_lat, max_lat,
                                        number_of_points_per_side)

        west_border[:, 0] = min_lng
        west_border[:, 1] = np.linspace(max_lat, min_lat,
                                        number_of_points_per_side)

        # Rotate everything.
        for border in [north_border, south_border, east_border, west_border]:
            for _i in xrange(number_of_points_per_side):
                border[_i, 1], border[_i, 0] = rotations.rotate_lat_lon(
                    border[_i, 1], border[_i, 0], rotation_axis,
                    rotation_angle_in_degree)

        border = np.concatenate([north_border, south_border, east_border,
                                 west_border])

        min_lng, max_lng = border[:, 0].min(), border[:, 0].max()
        min_lat, max_lat = border[:, 1].min(), border[:, 1].max()

        return min_lat, max_lat, min_lng, max_lng
