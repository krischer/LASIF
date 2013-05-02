#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple L2-norm misfit.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np


def l2NormMisfit(data, synthetic, component, axis=None):
    """
    Calculates the L2-norm misfit and adjoint source.

    :type data: np.ndarray
    :param data: The measured data array
    :type synthetic: np.ndarray
    :param synthetic: The calculated data array
    :type component: str
    :param component: The orientation of both the data and the synthetic trace.
        Has to be one of 'N', 'E', 'Z'.
    :param axis: matplotlib.axis
    :type axis: If given, a plot of the misfit will be drawn in axis. In this
        case this is just the squared difference between both traces.
    :rtype: (float, list of nd.arrays)
    :returns: Returns a tuple, with the first entry being the calculated
        misfit. The second entry is a list with 3 np.ndarrays containing the
        adjoint source for the (in this order) 'N', 'E', and 'Z' components.
    """
    if len(synthetic) != len(data):
        msg = "Both arrays need to have equal length"
        raise ValueError(msg)
    diff = synthetic - data
    diff = np.require(diff, dtype="float64")
    squared_diff = diff ** 2
    l2norm = np.sum(squared_diff)

    # Only the component of the data has non-zeros values for the adjoint
    # source.
    adjoint_source = (-1.0 * diff)[::-1]
    data_1 = np.zeros(len(adjoint_source), dtype=adjoint_source.dtype)
    data_2 = np.zeros(len(adjoint_source), dtype=adjoint_source.dtype)

    source = [data_1, data_2]
    if component == "N":
        source.insert(0, adjoint_source)
    elif component == "E":
        source.insert(1, adjoint_source)
    elif component == "Z":
        source.insert(2, adjoint_source)
    else:
        raise NotImplementedError

    if axis:
        axis.cla()
        axis.plot(squared_diff, color="black")
        axis.set_title("L2-Norm difference - Misfit: %e" % l2norm)
        axis.set_xlim(0, len(data))
        s_max = squared_diff.max()
        axis.set_ylim(-0.1 * s_max, 2.1 * s_max)
        axis.set_xticks([])
        #axis.set_yticks([])
        if not hasattr(axis, "twin_axis"):
            axis.twin_axis = axis.twinx()
        ax2 = axis.twin_axis
        ax2.plot(data, color="black", alpha=0.4)
        ax2.plot(synthetic, color="red", alpha=0.4)
        ax2.set_xlim(0, len(data))
        min_value = min(data.min(), synthetic.min())
        max_value = max(data.max(), synthetic.max())
        diff = max_value - min_value
        ax2.set_ylim(min_value - 1.1 * diff,  max_value + 0.1 * diff)
        ax2.set_xticks([])
        ax2.set_yticks([])

    return (l2norm, source)
