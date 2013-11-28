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

    :param npts: The desired number of samples.
    :param delta: The sample spacing.
    :param freqmin: The minimum desired frequency.
    :param freqmax:  The maximum desired frequency.
    """
    trace = obspy.Trace(data=np.zeros(npts * 3))
    trace.data[npts:] = 1.0
    trace.stats.delta = delta

    # Filter twice to get a sharper filter with less numerical errors. This has
    # the disadvantage of shifting the data in time so the same has to be done
    # to the synthetics.
    trace.filter("lowpass", freq=freqmax, corners=4)
    trace.taper(0.05, type="hann")
    trace.filter("highpass", freq=freqmin, corners=2)
    trace.taper(0.05, type="hann")

    trace.filter("lowpass", freq=freqmax, corners=4)
    trace.taper(0.05, type="hann")
    trace.filter("highpass", freq=freqmin, corners=2)
    trace.taper(0.05, type="hann")

    return trace.data[npts: npts * 2]
