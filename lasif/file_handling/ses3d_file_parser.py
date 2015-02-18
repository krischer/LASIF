#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A function reading SES3D output files an converting them to ObsPy Stream
objects.

Can be tied directly into ObsPy's plugin system by setting the correct entry
points in the setup.py.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012-2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
import warnings

from lasif import rotations


# The three different possibilities for the first line of a SES3D file. Used
# for identification purposes.
POSSIBLE_FIRST_LINES = [
    "theta component seismograms",
    "phi component seismograms",
    "r component seismograms"]

COMPONENT_MAP = {
    "theta": "X",
    "phi": "Y",
    "r": "Z"
}


def is_SES3D(filename_or_file_object):
    """
    Returns True if the file is a SES3D file, False otherwise.

    Works with filenames and file-like objects (open files, StringIO, ...).
    """
    opened_file = False
    if not hasattr(filename_or_file_object, "read"):
        filename_or_file_object = open(filename_or_file_object, "r")
        opened_file = True
    try:
        first_line = filename_or_file_object.readline()
    except:
        if opened_file:
            filename_or_file_object.close()
        return False
    if opened_file:
        filename_or_file_object.close()
    first_line = first_line.strip()
    if first_line in POSSIBLE_FIRST_LINES:
        return True
    return False


def read_SES3D(file_or_file_object, headonly=False, *args, **kwargs):
    """
    Turns a SES3D file into a obspy.core.Stream object.

    SES3D files do not contain a starttime and thus the first first sample will
    always begin at 1970-01-01T00:00:00.

    The data will be a floating point array of the ground velocity in meters
    per second.

    Furthermore every trace will have a trace.stats.ses3d dictionary which
    contains the following six keys:
        * receiver_latitude
        * receiver_longitde
        * receiver_depth_in_m
        * source_latitude
        * source_longitude
        * source_depth_in_m

    The network, station, and location attributes of the trace will be empty,
    and the channel will be set to either 'X' (south component), 'Y' (east
    component), or 'Z' (vertical component).
    """
    if not hasattr(file_or_file_object, "read"):
        with open(file_or_file_object, "rb") as fh:
            return _read_SES3D(fh, headonly=headonly)
    else:
        return _read_SES3D(file_or_file_object, headonly=headonly)


def _read_SES3D(fh, headonly=False):
    """
    Internal SES3D parsing routine.
    """
    # Import here to avoid circular imports.
    from obspy.core import AttribDict, Trace, Stream

    # Read the header.
    component = fh.readline().split()[0].lower()
    npts = int(fh.readline().split()[-1])
    delta = float(fh.readline().split()[-1])
    # Skip receiver location line.
    fh.readline()
    rec_loc = fh.readline().split()
    rec_x, rec_y, rec_z = map(float, [rec_loc[1], rec_loc[3], rec_loc[5]])
    # Skip the source location line.
    fh.readline()
    src_loc = fh.readline().split()
    src_x, src_y, src_z = map(float, [src_loc[1], src_loc[3], src_loc[5]])

    # Read the data.
    if headonly is False:
        data = np.array(map(float, fh.readlines()),
                        dtype="float32")
    else:
        data = np.array([])

    ses3d = AttribDict()
    ses3d.receiver_latitude = rotations.colat2lat(rec_x)
    ses3d.receiver_longitude = rec_y
    ses3d.receiver_depth_in_m = rec_z
    ses3d.source_latitude = rotations.colat2lat(src_x)
    ses3d.source_longitude = src_y
    ses3d.source_depth_in_m = src_z

    header = {
        "delta": delta,
        "channel": COMPONENT_MAP[component],
        "ses3d": ses3d,
        "npts": npts
    }

    # Setup Obspy Stream/Trace structure.
    tr = Trace(data=data, header=header)

    # Small check.
    if headonly is False and npts != tr.stats.npts:
        msg = "The sample count specified in the header does not match " + \
            "the actual data count."
        warnings.warn(msg)
    return Stream(traces=[tr])
