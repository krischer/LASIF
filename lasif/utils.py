#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some utility functionality.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from collections import namedtuple
from geographiclib import geodesic
from fnmatch import fnmatch

from lasif import LASIFNotFoundError


def is_mpi_env():
    """
    Returns True if currently in an MPI environment.
    """
    from mpi4py import MPI
    if MPI.COMM_WORLD.size == 1 and MPI.COMM_WORLD.rank == 0:
        return False
    return True


def channel_in_parser(parser_object, channel_id, starttime, endtime):
    """
    Simply function testing if a given channel is part of a Parser object.

    Returns True or False.

    :type parser_object: :class:`obspy.io.xseed.Parser`
    :param parser_object: The parser object.
    """
    channels = parser_object.get_inventory()["channels"]
    for chan in channels:
        if not fnmatch(chan["channel_id"], channel_id):
            continue
        if starttime < chan["start_date"]:
            continue
        if chan["end_date"] and \
                (endtime > chan["end_date"]):
            continue
        return True
    return False


def table_printer(header, data):
    """
    Pretty table printer.

    :type header: A list of strings
    :param data: A list of lists containing data items.
    """
    row_format = "{:>15}" * (len(header))
    print(row_format.format(*(["=" * 15] * len(header))))
    print(row_format.format(*header))
    print(row_format.format(*(["=" * 15] * len(header))))
    for row in data:
        print(row_format.format(*row))


def sizeof_fmt(num):
    """
    Handy formatting for human readable filesize.

    From http://stackoverflow.com/a/1094933/1657047
    """
    for x in ["bytes", "KB", "MB", "GB"]:
        if num < 1024.0 and num > -1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, "TB")


Point = namedtuple("Point", ["lat", "lng"])


def greatcircle_points(point_1, point_2, max_extension=None,
                       max_npts=3000):
    """
    Generator yielding a number points along a greatcircle from point_1 to
    point_2. Max extension is the normalization factor. If the distance between
    point_1 and point_2 is exactly max_extension, then 3000 points will be
    returned, otherwise a fraction will be returned.

    If max_extension is not given, the generator will yield exactly max_npts
    points.
    """
    point = geodesic.Geodesic.WGS84.Inverse(
        lat1=point_1.lat, lon1=point_1.lng, lat2=point_2.lat,
        lon2=point_2.lng)
    line = geodesic.Geodesic.WGS84.Line(
        point_1.lat, point_1.lng, point["azi1"])

    if max_extension:
        npts = int((point["a12"] / float(max_extension)) * max_npts)
    else:
        npts = max_npts - 1
    if npts == 0:
        npts = 1
    for i in range(npts + 1):
        line_point = line.Position(i * point["s12"] / float(npts))
        yield Point(line_point["lat2"], line_point["lon2"])


def channel2station(value):
    """
    Helper function converting a channel id to a station id. Will not change
    a passed station id.

    :param value: The channel id as a string.

    >>> channel2station("BW.FURT.00.BHZ")
    'BW.FURT'
    >>> channel2station("BW.FURT")
    'BW.FURT'
    """
    return ".".join(value.split(".")[:2])


def select_component_from_stream(st, component):
    """
    Helper function selecting a component from a Stream an raising the proper
    error if not found.

    This is a bit more flexible then stream.select() as it works with single
    letter channels and lowercase channels.
    """
    component = component.upper()
    component = [tr for tr in st if tr.stats.channel[-1].upper() == component]
    if not component:
        raise LASIFNotFoundError("Component %s not found in Stream." %
                                 component)
    elif len(component) > 1:
        raise LASIFNotFoundError("More than 1 Trace with component %s found "
                                 "in Stream." % component)
    return component[0]


def get_event_filename(event, prefix):
    """
    Helper function generating a descriptive event filename.

    :param event: The event object.
    :param prefix: A prefix for the file, denoting e.g. the event catalog.

    >>> from obspy import read_events
    >>> event = read_events()[0]
    >>> print(get_event_filename(event, "GCMT"))
    GCMT_event_KYRGYZSTAN-XINJIANG_BORDER_REG._Mag_4.4_2012-4-4-14.h5
    """
    from obspy.geodetics import FlinnEngdahl

    mag = event.preferred_magnitude() or event.magnitudes[0]
    org = event.preferred_origin() or event.origins[0]

    # Get the flinn_engdahl region for a nice name.
    fe = FlinnEngdahl()
    region_name = fe.get_region(org.longitude, org.latitude)
    region_name = region_name.replace(" ", "_")
    # Replace commas, as some file systems cannot deal with them.
    region_name = region_name.replace(",", "")

    return "%s_event_%s_Mag_%.1f_%s-%s-%s-%s.h5" % \
        (prefix, region_name, mag.mag, org.time.year, org.time.month,
         org.time.day, org.time.hour)
