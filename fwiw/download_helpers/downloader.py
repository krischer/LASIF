#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some convenience function to download waveform and station data.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/licenses/gpl.html)
"""
import colorama
from datetime import datetime
import logging
import numpy as np
import obspy
import os
import sys

from fwiw import rotations
from fwiw.download_helpers.availability import get_availability
import fwiw.download_helpers.waveforms
import fwiw.download_helpers.stations


class Logger(object):
    """
    Simple logging class printing to the screen in color as well as to a file.
    """
    def __init__(self, log_filename, debug=False):
        FORMAT = "[%(asctime)-15s] %(levelname)s: %(message)s"
        logging.basicConfig(filename=log_filename, level=logging.DEBUG,
            format=FORMAT)
        self.logger = logging.getLogger("FWIW")
        self.set_debug(debug)

    def set_debug(self, value):
        if value:
            self._debug = True
            self.logger.setLevel(logging.DEBUG)
        else:
            self._debug = False
            self.logger.setLevel(logging.INFO)

    def critical(self, msg):
        print(colorama.Fore.WHITE + colorama.Back.RED +
            self._format_message("CRITICAL", msg) + colorama.Style.RESET_ALL)
        self.logger.critical(msg)

    def error(self, msg):
        print(colorama.Fore.RED + self._format_message("ERROR", msg) +
            colorama.Style.RESET_ALL)
        self.logger.error(msg)

    def warning(self, msg):
        print(colorama.Fore.YELLOW + self._format_message("WARNING", msg) +
            colorama.Style.RESET_ALL)
        self.logger.warning(msg)

    def info(self, msg):
        print(self._format_message("INFO", msg))
        self.logger.info(msg)

    def debug(self, msg):
        if not self._debug:
            return
        print(colorama.Fore.BLUE + self._format_message("DEBUG", msg) +
            colorama.Style.RESET_ALL)
        self.logger.debug(msg)

    def _format_message(self, prefix, msg):
        return "[%s] %s: %s" % (datetime.now(), prefix, msg)


def _get_maximum_bounds(min_lat, max_lat, min_lng, max_lng, rotation_axis,
        rotation_angle_in_degree):
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


def download_waveforms(min_latitude, max_latitude, min_longitude,
        max_longitude, rotation_axis, rotation_angle_in_degree, starttime,
        endtime, arclink_user, channel_priority_list, logfile, download_folder,
        waveform_format="mseed"):
    """
    Convenience function downloading all waveform files in the specified
    spherical section domain.
    """
    # Init logger.
    logger = Logger(log_filename=logfile, debug=True)

    # Log some basic information
    logger.info(70 * "=")
    logger.info(70 * "=")
    logger.info("Starting waveform downloads...")

    def filter_location_fct(latitude, longitude):
        """
        Simple function checking if a geographic point is placed inside a
        rotated spherical section. It simple rotates the point and checks if it
        is inside the unrotated domain. The domain specification are passed in
        as a closure.

        Returns True or False.
        """
        # Rotate the station and check if it is still in bounds.
        r_lat, r_lng = rotations.rotate_lat_lon(latitude, longitude,
            rotation_axis, -1.0 * rotation_angle_in_degree)
        # Check if in bounds. If not continue.
        if not (min_latitude <= r_lat <= max_latitude) or \
                not (min_longitude <= r_lng <= max_longitude):
            return False
        return True

    # Get the maximum bounds of the domain.
    min_lat, max_lat, min_lng, max_lng = _get_maximum_bounds(min_latitude,
        max_latitude, min_longitude, max_longitude, rotation_axis,
        rotation_angle_in_degree)

    # Get the availability.
    channels = get_availability(min_lat, max_lat, min_lng, max_lng, starttime,
        endtime, arclink_user, channel_priority_list=channel_priority_list,
        logger=logger, filter_location_fct=filter_location_fct)

    if not channels:
        msg = "No matching channels found. Program will terminate."
        logger.critical(msg)
        sys.exit(1)

    def get_channel_filename(channel_id):
        """
        Get the filename given a seed channel id.

        File format and other things will be available via closures.
        """
        filename = "%s.%s" % (channel_id, waveform_format.lower())
        return os.path.join(download_folder, filename)

    # Now get a list of those channels that still need downloading.
    channels_to_download = []
    for chan in channels.iterkeys():
        filename = get_channel_filename(chan)
        if os.path.exists(filename):
            continue
        channels_to_download.append(chan)

    def save_channel(trace):
        """
        Save a single trace.
        """
        filename = get_channel_filename(trace.id)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        try:
            trace.write(filename, format=waveform_format)
        except Exception as e:
            msg = "Error writing Trace. This is likely due to corrupted data."
            msg += "Error message: %s" % e.message
            logger.error(msg)

    logger.info("Attempting to download %i (missing) waveform channels..." %
        len(channels_to_download))
    # Download in chunks of 100 channels.
    channels_to_download.sort()
    CHUNK_SIZE = 100
    current_position = 1
    successful_downloads = 0
    for chunk in (channels_to_download[_i: _i + CHUNK_SIZE] for _i in xrange(0,
            len(channels_to_download), CHUNK_SIZE)):
        logger.info(70 * "=")
        logger.info("Starting download for channels %i to %i..." %
            (current_position, current_position + CHUNK_SIZE))
        logger.info(70 * "=")
        current_position += CHUNK_SIZE
        successful_downloads += \
            fwiw.download_helpers.waveforms.download_waveforms(
                chunk, starttime, endtime, 0.95, save_trace_fct=save_channel,
                arclink_user=arclink_user, logger=logger)

    print "Done. Successfully downloaded %i waveform channels." % \
        successful_downloads


def download_stations(channels, resp_file_folder, station_xml_folder,
        dataless_seed_folder, logfile, arclink_user, has_station_file_fct,
        get_station_filename_fct):
    """
    Convenience function downloading station information for all channels in
    the channels filename list. It will only download what does not exist yet.

    It will first attempt to download datalessSEED and RESP files from ArcLink
    and then, do the same for all missing data from IRIS, except it will
    attempt to download StationXML and RESP files.

    :param channels: A list of filenames pointing to all channels.
    :param resp_file_folder: The folder where the RESP file are stored.
    :param station_XML: The folder where the StationXML files are stored.
    :param dataless_seed_folder: The folder where the dataless SEED files are
        stored.

    :param has_station_file_fct: Function has_station_file_fct(filename)
        returning True or False if the station file for a given waveform file
        path already exists.
    :param get_station_filename_fct: Function get_station_filename_fct(network,
        station, location, channel, format) the simply returns the path where
        the given file should be written to.
    """
    # Init logger.
    logger = Logger(log_filename=logfile, debug=False)

    # Log some basic information
    logger.info(70 * "=")
    logger.info(70 * "=")
    logger.info("Starting to download %i station files..." % len(channels))

    missing_files = []

    # First figure out what data is still needed.
    for filename in channels:
        if has_station_file_fct(filename):
            continue
        tr = obspy.read(filename)[0]
        missing_files.append({"network": tr.stats.network,
            "station": tr.stats.station,
            "location": tr.stats.location,
            "channel": tr.stats.channel,
            "starttime": tr.stats.starttime,
            "endtime": tr.stats.endtime})

    existing_files = len(channels) - len(missing_files)
    logger.info("%i files already existing. They will skipped." %
        existing_files)
    logger.info("Starting download of %i missing files..." %
        len(missing_files))

    def save_station_file(memfile, network, station, location, channel,
            format):
        """
        Callback function saving a single file given as a StringIO instance.
        """
        filename = get_station_filename_fct(network, station, location,
            channel, format)
        memfile.seek(0, 0)
        with open(filename, "wb") as open_file:
            open_file.write(memfile.read())

    # Now download all the missing stations files.
    successful_downloads = \
        fwiw.download_helpers.stations.download_station_files(missing_files,
            save_station_fct=save_station_file, arclink_user=arclink_user,
            logger=logger)

    print "Done. Successfully downloaded %i station files." % \
        successful_downloads
