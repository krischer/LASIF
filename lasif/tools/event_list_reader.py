#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some tools to help transitioning from the event list format previously in use
for running SES3D inversions to QuakeML files.

Probably not generally useful.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude, \
    FocalMechanism, MomentTensor, Tensor
import os


def read_event_list(filename):
    """
    Reads a certain event list format.
    """
    if not os.path.exists(filename):
        msg = "Can not find file %s." % filename
        raise Exception(msg)
    events = {}
    with open(filename, "rU") as open_file:
        for line in open_file:
            line = line.strip()
            line = line.split()
            if len(line) < 14:
                continue
            if not line[0].isdigit():
                continue
            index, date, colat, lon, depth, exp, Mrr, Mtt, Mpp, Mrt, Mrp, \
                Mtp, time, Mw = line[:14]
            index, exp = map(int, (index, exp))
            colat, lon, depth, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, Mw = map(
                float, (colat, lon, depth, Mrr, Mtt, Mpp, Mrt, Mrp, Mtp, Mw))
            year, month, day = map(int, date.split("/"))
            split_time = time.split(":")
            if len(split_time) == 3:
                hour, minute, second = split_time
                second, microsecond = second.split(".")
            elif len(split_time) == 4:
                hour, minute, second, microsecond = split_time
            else:
                raise NotImplementedError
            microsecond = int(microsecond) * 10 ** (6 - len(microsecond))
            hour, minute, second = map(int, (hour, minute, second))
            event_time = UTCDateTime(year, month, day, hour, minute, second,
                                     microsecond)
            event = {
                "longitude": lon,
                "latitude": -1.0 * (colat - 90.0),
                "depth_in_km": depth,
                "time": event_time,
                "identifier": index,
                "Mw": Mw,
                "Mrr": Mrr * 10 ** exp,
                "Mtt": Mtt * 10 ** exp,
                "Mpp": Mpp * 10 ** exp,
                "Mrt": Mrt * 10 ** exp,
                "Mrp": Mrp * 10 ** exp,
                "Mtp": Mtp * 10 ** exp}
            events[index] = event
    return events


def event_list_to_quakeml(filename, folder):
    """
    Helper function to convert all events in an event file to QuakeML.
    """
    events = read_event_list(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for key, value in events.iteritems():
        filename = os.path.join(folder, "%s%sxml" % (key, os.path.extsep))
        event_to_quakeml(value, filename)


def event_to_quakeml(event, filename):
    """
    Write one of those events to QuakeML.
    """
    # Create all objects.
    cat = Catalog()
    ev = Event()
    org = Origin()
    mag = Magnitude()
    fm = FocalMechanism()
    mt = MomentTensor()
    t = Tensor()
    # Link them together.
    cat.append(ev)
    ev.origins.append(org)
    ev.magnitudes.append(mag)
    ev.focal_mechanisms.append(fm)
    fm.moment_tensor = mt
    mt.tensor = t

    # Fill values
    ev.resource_id = "smi:inversion/%s" % str(event["identifier"])
    org.time = event["time"]
    org.longitude = event["longitude"]
    org.latitude = event["latitude"]
    org.depth = event["depth_in_km"] * 1000

    mag.mag = event["Mw"]
    mag.magnitude_type = "Mw"

    t.m_rr = event["Mrr"]
    t.m_tt = event["Mpp"]
    t.m_pp = event["Mtt"]
    t.m_rt = event["Mrt"]
    t.m_rp = event["Mrp"]
    t.m_tp = event["Mtp"]

    cat.write(filename, format="quakeml")
