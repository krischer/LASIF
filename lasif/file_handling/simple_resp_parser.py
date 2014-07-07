#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some basic utility to parse RESP files.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from obspy import UTCDateTime

from lasif import LASIFError


def get_inventory(resp_file, remove_duplicates=False):
    """
    Simple function reading a RESP file and returning a list of dictionaries.
    Each dictionary contains the following keys for each channel found in the
    RESP file:

        * network
        * station
        * location
        * channel
        * start_date
        * end_date
        * channel_id

    :param resp_file: Resp file to open.
    :param remove_duplicates: Some RESP files contain the same values twice.
        This option the duplicates. Defaults to False.
    """
    channels = []
    with open(resp_file, "rU") as open_file:
        current_channel = {}
        for line in open_file:
            line = line.strip().upper()
            if line.startswith("B050F03"):
                current_channel["station"] = line.split()[-1]
                if _is_channel_complete(current_channel):
                    channels.append(current_channel)
                    current_channel = {}
            elif line.startswith("B050F16"):
                current_channel["network"] = line.split()[-1]
                if _is_channel_complete(current_channel):
                    channels.append(current_channel)
                    current_channel = {}
            elif line.startswith("B052F03"):
                location = line.split()[-1]
                if location == "??":
                    location = ""
                current_channel["location"] = location
                if _is_channel_complete(current_channel):
                    channels.append(current_channel)
                    current_channel = {}
            elif line.startswith("B052F04"):
                current_channel["channel"] = line.split()[-1]
                if _is_channel_complete(current_channel):
                    channels.append(current_channel)
                    current_channel = {}
            elif line.startswith("B052F22"):
                current_channel["start_date"] = _parse_resp_datetime_string(
                    line.split()[-1])
                if _is_channel_complete(current_channel):
                    channels.append(current_channel)
                    current_channel = {}
            elif line.startswith("B052F23"):
                current_channel["end_date"] = _parse_resp_datetime_string(
                    line.split()[-1])
                if _is_channel_complete(current_channel):
                    channels.append(current_channel)
                    current_channel = {}
    for channel in channels:
        channel["channel_id"] = "{network}.{station}.{location}.{channel}"\
            .format(**channel)
    # Make unique list if requested.
    if remove_duplicates is True:
        unique_list = []
        for channel in channels:
            if channel in unique_list:
                continue
            unique_list.append(channel)
        channels = unique_list
    if not channels:
        raise LASIFError("'%s' is not a valid RESP file." % resp_file)
    return channels


def _is_channel_complete(channel_dict):
    keys = sorted(["station", "network", "location", "channel",
                   "start_date", "end_date"])
    if sorted(channel_dict.keys()) == keys:
        return True
    return False


def _parse_resp_datetime_string(datetime_string):
    """
    Helper method to parse the different datetime strings.
    """
    if datetime_string == "TIME":
        return None
    dt = datetime_string.split(",")
    # Parse 2003,169
    if len(dt) == 2:
        year, julday = map(int, dt)
        return UTCDateTime(year=year, julday=julday)
    # Parse 2003,169,00:00:00.0000
    elif len(dt) == 3:
        year, julday = map(int, dt[:2])
        time_split = dt[-1].split(":")
        if len(time_split) == 3:
            hour, minute, second = time_split
            # Add the seconds seperately because the constructor does not
            # accept seconds as floats.
            return UTCDateTime(year=year, julday=julday, hour=int(hour),
                               minute=int(minute)) + float(second)
        elif len(time_split) == 2:
            hour, minute = map(int, time_split)
            return UTCDateTime(year=year, julday=julday, hour=int(hour),
                               minute=int(minute))
        elif len(time_split) == 1:
            hour = int(time_split[0])
            return UTCDateTime(year=year, julday=julday, hour=hour)
        else:
            msg = "Unknown datetime representation %s" % datetime_string
            raise NotImplementedError(msg)
    else:
        msg = "Unknown datetime representation %s" % datetime_string
        raise NotImplementedError(msg)
