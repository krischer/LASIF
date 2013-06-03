#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File containing different source time functions.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import obspy
import numpy as np


def filtered_heaviside(npts, delta, freqmin, freqmax):
    """
    Filtered Heaviside source time function.

    Freqmin and freqmax specify the bandpass filter frequencies.
    """
    trace = obspy.Trace(data=np.ones(npts))
    trace.stats.delta = delta
    trace.filter("lowpass", freq=freqmax, corners=5)
    trace.filter("highpass", freq=freqmin, corners=2)

    return trace.data
