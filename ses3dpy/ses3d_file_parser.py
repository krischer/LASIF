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
from obspy.core import AttribDict, Trace, Stream
import warnings

import rotations


# The three different possibilities for the first line of a SES3D file. Used
# for identification purposes.
POSSIBLE_FIRST_LINES = [ \
    "theta component seismograms",
    "phi component seismograms",
    "r component seismograms"]


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


def read_SES3D(filename, *args, **kwargs):
    with open(filename, "r") as open_file:
        component = open_file.readline().split()[0].lower()
        npts = int(open_file.readline().split()[-1])
        delta = float(open_file.readline().split()[-1])
        # Skip receiver location line.
        open_file.readline()
        rec_loc = open_file.readline().split()
        rec_x, rec_y, rec_z = map(float, [rec_loc[1], rec_loc[3], rec_loc[5]])
        # Skip the source location line.
        open_file.readline()
        src_loc = open_file.readline().split()
        src_x, src_y, src_z = map(float, [src_loc[1], src_loc[3], src_loc[5]])
        # The rest is data.
        data = np.array(map(float, open_file.readlines()), dtype="float32")
    tr = Trace(data=data)
    tr.stats.delta = delta
    # Invert theta components to let them point north.
    if component == "theta":
        tr.data *= -1.0
    # Map the channel attributes.
    tr.stats.channel = {"theta": "N",
        "phi": "E",
        "r": "Z"}[component]
    tr.stats.ses3d = AttribDict()
    tr.stats.ses3d.receiver_latitude = - (rec_x - 90.0)
    tr.stats.ses3d.receiver_longitude = rec_y
    tr.stats.ses3d.receiver_depth_in_m = rec_z
    tr.stats.ses3d.source_latitude = - (src_x - 90.0)
    tr.stats.ses3d.source_longitude = src_y
    tr.stats.ses3d.source_depth_in_m = src_z
    # Small check.
    if npts != tr.stats.npts:
        msg = "The sample count specified in the header does not match " + \
            "the actual data count."
        warnings.warn(msg)
    return Stream(traces=[tr])
