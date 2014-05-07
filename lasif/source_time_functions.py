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
import numpy as np
import obspy.signal.filter


def filtered_heaviside(npts, delta, freqmin, freqmax):
    """
    Filtered Heaviside source time function.

    Freqmin and freqmax specify the bandpass filter frequencies.

    :param npts: The desired number of samples.
    :param delta: The sample spacing.
    :param freqmin: The minimum desired frequency.
    :param freqmax:  The maximum desired frequency.
    """
    heaviside = np.ones(npts)

    # Apply ObsPy filters.
    heaviside = obspy.signal.filter.highpass(heaviside, freqmin, 1.0 / delta,
                                             2, zerophase=False)
    heaviside = obspy.signal.filter.lowpass(heaviside, freqmax, 1.0 / delta,
                                            5, zerophase=False)

    return heaviside
