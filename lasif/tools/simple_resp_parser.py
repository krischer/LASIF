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


def get_inventory(resp_file):
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

    :param resp_file: Resp file to open.
    """
    channels = []
    with open(resp_file, "rt") as open_file:
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
        else:
            msg = "Unknown datetime representation %s" % datetime_string
            raise NotImplementedError(msg)
    else:
        msg = "Unknown datetime representation %s" % datetime_string
        raise NotImplementedError(msg)
