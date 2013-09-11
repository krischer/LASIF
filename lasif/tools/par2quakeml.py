#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert old SES3D Par files to QuakeML

**Usage:**

Still to be specified.

:copyright:
    Andreas Fichtner (fichtner@erdw.ethz.ch), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

import sys
sys.path.append('../')
import lasif.rotations as rot
import numpy as np
from obspy.core import UTCDateTime
from obspy.core.event import Catalog, Magnitude, Event, Origin, \
    FocalMechanism, Tensor, MomentTensor, NodalPlanes


def par2quakeml(Par_filename, QuakeML_filename, rotation_axis=[0.0, 1.0, 0.0],
                rotation_angle=-57.5, origin_time="2000-01-01 00:00:00.0",
                event_type="other event"):
    # initialise event
    ev = Event()

    # open and read Par file
    fid = open(Par_filename, 'r')

    fid.readline()
    fid.readline()
    fid.readline()
    fid.readline()

    lat_old = 90.0 - float(fid.readline().strip().split()[0])
    lon_old = float(fid.readline().strip().split()[0])
    depth = float(fid.readline().strip().split()[0])

    fid.readline()

    Mtt_old = float(fid.readline().strip().split()[0])
    Mpp_old = float(fid.readline().strip().split()[0])
    Mrr_old = float(fid.readline().strip().split()[0])
    Mtp_old = float(fid.readline().strip().split()[0])
    Mtr_old = float(fid.readline().strip().split()[0])
    Mpr_old = float(fid.readline().strip().split()[0])

    # rotate event into physical domain

    lat, lon = rot.rotate_lat_lon(lat_old, lon_old, rotation_axis,
                                  rotation_angle)
    Mrr, Mtt, Mpp, Mtr, Mpr, Mtp = rot.rotate_moment_tensor(
        Mrr_old, Mtt_old, Mpp_old, Mtr_old, Mpr_old, Mtp_old, lat_old, lon_old,
        rotation_axis, rotation_angle)

    # populate event origin data
    ev.event_type = event_type

    ev_origin = Origin()
    ev_origin.time = UTCDateTime(origin_time)
    ev_origin.latitude = lat
    ev_origin.longitude = lon
    ev_origin.depth = depth
    ev.origins.append(ev_origin)

    # populte event moment tensor

    ev_tensor = Tensor()
    ev_tensor.m_rr = Mrr
    ev_tensor.m_tt = Mtt
    ev_tensor.m_pp = Mpp
    ev_tensor.m_rt = Mtr
    ev_tensor.m_rp = Mpr
    ev_tensor.m_tp = Mtp

    ev_momenttensor = MomentTensor()
    ev_momenttensor.tensor = ev_tensor
    ev_momenttensor.scalar_moment = np.sqrt(Mrr ** 2 + Mtt ** 2 + Mpp ** 2 +
                                            Mtr ** 2 + Mpr ** 2 + Mtp ** 2)

    ev_focalmechanism = FocalMechanism()
    ev_focalmechanism.moment_tensor = ev_momenttensor
    ev_focalmechanism.nodal_planes = NodalPlanes().setdefault(0, 0)

    ev.focal_mechanisms.append(ev_focalmechanism)

    # populate event magnitude
    ev_magnitude = Magnitude()
    ev_magnitude.mag = 0.667 * (np.log10(ev_momenttensor.scalar_moment) - 9.1)
    ev_magnitude.magnitude_type = 'Mw'
    ev.magnitudes.append(ev_magnitude)

    # write QuakeML file
    cat = Catalog()
    cat.append(ev)
    cat.write(QuakeML_filename, format="quakeml")

    # clean up
    fid.close()
