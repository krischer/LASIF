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
    trace = obspy.Trace(data=np.ones(npts))
    trace.stats.delta = delta
    # The number of corners have been empirically determined to yield a nice
    # source time function. The lowpass is much sharper to ensure that no
    # energy leaks above the maximum frequency.
    trace.filter("lowpass", freq=freqmax, corners=5)
    trace.filter("highpass", freq=freqmin, corners=2)

    return trace.data
