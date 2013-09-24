#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple tests for the constant Q model calculator.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from lasif.tools.Q_discrete import calculate_Q_model
import numpy as np


def test_weights_and_relaxation_times():
    """
    Regression test for the weights and relaxation times.
    """
    # The last two return values are the frequency dependent Q factor and
    # the phase velocity. Useful for plotting but not here.
    weights, relaxation_times, _, _ = calculate_Q_model(
        Q=100.0,
        N=3,
        f_min=1.0 / 100.0,
        f_max=1.0 / 10.0,
        max_iterations=10000,
        initial_temperature=0.1,
        cooling_factor=0.9998)
    np.testing.assert_arrays_almost_equal(
        weights, [2.51074556, 2.46112185, 0.05143481])
    np.testing.assert_arrays_almost_equal(
        relaxation_times, [1.73495054, 14.47838511, 20.62732302])
