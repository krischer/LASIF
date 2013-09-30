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
import random
import numpy as np

from lasif.tools.Q_discrete import calculate_Q_model


def test_weights_and_relaxation_times():
    """
    Regression test for the weights and relaxation times.
    """
    # Set the seed to get reproducible results.
    random.seed(12345)

    # These are the D_p and tau_p, respectively.
    weights, relaxation_times, = calculate_Q_model(
        Q=100.0,
        N=3,
        f_min=1.0 / 100.0,
        f_max=1.0 / 10.0,
        max_iterations=10000,
        initial_temperature=0.1,
        cooling_factor=0.9998)

    np.testing.assert_array_almost_equal(
        weights, np.array([2.50960201, 2.31899515, 0.19681762]))
    np.testing.assert_array_almost_equal(
        relaxation_times, np.array([1.73160984, 14.41562154, 16.70330157]))
