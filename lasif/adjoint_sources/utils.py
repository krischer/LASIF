#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some functionality useful for calculating the adjoint sources.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np


def matlab_range(start, stop, step):
    """
    Simple function emulating the behaviour of Matlab's colon notation.

    This is very similar to np.arange(), except that the endpoint is included
    if it would be the logical next sample. Useful for translating Matlab code
    to Python.
    """
    # Some tolerance
    if (abs(stop - start) / step) % 1 < 1E-7:
        return np.arange(start, stop + step / 2.0, step)
    return np.arange(start, stop, step)
