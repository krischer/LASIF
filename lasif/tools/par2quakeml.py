#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert old SES3D Par files to QuakeML

**Usage:**

Input filenames of Par file and the desired filename of the output QuakeML
file, including the full path.


:copyright:
    Andreas Fichtner (fichtner@erdw.ethz.ch), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
from obspy.core import UTCDateTime
from obspy.core.event import Catalog, Magnitude, Event, Origin, \
    FocalMechanism, Tensor, MomentTensor, NodalPlanes

import lasif.rotations as rot


def par2quakeml(par_filename, output_filename, rotation_axis=[0.0, 1.0, 0.0],
        rotation_angle=-57.5, origin_time="2000-01-01 00:00:00.0"):

    # Open file and extract coordinates and moment tensor values
    with open(par_filename, "rt") as fp:
        lines = [_i.split()[0] for _i in fp.readlines()]
    colat, lon, depth = map(float, lines[4:7])
    lat = rot.colat2lat(colat)
    mtt, mpp, mrr, mtp, mtr, mpr = map(float, lines[8:14])

    # Rotate into physical domain
    lat, lon = rot.rotate_lat_lon(lat, lon, rotation_axis, rotation_angle)
    mrr, mtt, mpp, mtr, mpr, mtp = rot.rotate_moment_tensor(mrr, mtt, mpp, mtr,
        mpr, mtp, lat, lon, rotation_axis, rotation_angle)

    # Initialise event
    ev = Event(event_type="earthquake")

    ev_origin = Origin(time=UTCDateTime(origin_time), latitude=lat,
        longitude=lon, depth=depth)
    ev.origins.append(ev_origin)

    # populte event moment tensor
    ev_tensor = Tensor(m_rr=mrr, m_tt=mtt, m_pp=mpp, m_rt=mtr, m_rp=mpr,
        m_tp=mtp)

    ev_momenttensor = MomentTensor(tensor=ev_tensor)
    ev_momenttensor.scalar_moment = np.sqrt(mrr ** 2 + mtt ** 2 + mpp ** 2 +
        mtr ** 2 + mpr ** 2 + mtp ** 2)

    ev_focalmechanism = FocalMechanism(moment_tensor=ev_momenttensor)
    ev_focalmechanism.nodal_planes = NodalPlanes().setdefault(0, 0)

    ev.focal_mechanisms.append(ev_focalmechanism)

    # populate event magnitude
    ev_magnitude = Magnitude()
    ev_magnitude.mag = 0.667 * (np.log10(ev_momenttensor.scalar_moment) - 9.1)
    ev_magnitude.magnitude_type = 'Mw'
    ev.magnitudes.append(ev_magnitude)

    # write QuakeML file
    cat = Catalog(events=[ev])
    cat.write(output_filename, format="quakeml", validate=True)
