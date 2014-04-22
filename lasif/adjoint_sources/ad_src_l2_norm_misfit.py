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


def adsrc_l2_norm_misfit(data, synthetic, axis=None):
    """
    Calculates the L2-norm misfit and adjoint source.

    :type data: np.ndarray
    :param data: The measured data array
    :type synthetic: np.ndarray
    :param synthetic: The calculated data array
    :param axis: matplotlib.axis
    :type axis: If given, a plot of the misfit will be drawn in axis. In this
        case this is just the squared difference between both traces.
    :rtype: dictionary
    :returns: Return a dictionary with three keys:
        * adjoint_source: The calculated adjoint source as a numpy array
        * misfit: The misfit value
        * messages: A list of strings giving additional hints to what happened
            in the calculation.
    """
    if len(synthetic) != len(data):
        msg = "Both arrays need to have equal length"
        raise ValueError(msg)
    diff = synthetic - data
    diff = np.require(diff, dtype="float64")
    squared_diff = diff ** 2
    l2norm = np.sum(squared_diff)

    adjoint_source = (-1.0 * diff)[::-1]

    if axis:
        axis.cla()
        axis.plot(squared_diff, color="black")
        axis.set_title("L2-Norm difference - Misfit: %e" % l2norm)
        axis.set_xlim(0, len(data))
        s_max = squared_diff.max()
        axis.set_ylim(-0.1 * s_max, 2.1 * s_max)
        axis.set_xticks([])
        # axis.set_yticks([])
        if not hasattr(axis, "twin_axis"):
            axis.twin_axis = axis.twinx()
        ax2 = axis.twin_axis
        ax2.plot(data, color="black", alpha=0.4)
        ax2.plot(synthetic, color="red", alpha=0.4)
        ax2.set_xlim(0, len(data))
        min_value = min(data.min(), synthetic.min())
        max_value = max(data.max(), synthetic.max())
        diff = max_value - min_value
        ax2.set_ylim(min_value - 1.1 * diff, max_value + 0.1 * diff)
        ax2.set_xticks([])
        ax2.set_yticks([])

    ret_dict = {
        "adjoint_source": adjoint_source,
        "misfit": l2norm,
        "messages": []}
    return ret_dict
