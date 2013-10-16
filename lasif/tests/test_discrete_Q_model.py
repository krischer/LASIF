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
import matplotlib as mpl
mpl.use("agg")

import numpy as np
import random

from lasif.tools import Q_discrete
from lasif.tests.testing_helpers import images_are_identical, \
    reset_matplotlib


def test_weights_and_relaxation_times():
    """
    Regression test for the weights and relaxation times.
    """
    # Set the seed to get reproducible results.
    random.seed(12345)

    # These are the D_p and tau_p, respectively.
    weights, relaxation_times, = Q_discrete.calculate_Q_model(
        N=3,
        f_min=1.0 / 100.0,
        f_max=1.0 / 10.0,
        iterations=10000,
        initial_temperature=0.1,
        cooling_factor=0.9998)

    np.testing.assert_array_almost_equal(
        weights, np.array([2.50960201, 2.31899515, 0.19681762]))
    np.testing.assert_array_almost_equal(
        relaxation_times, np.array([1.73160984, 14.41562154, 16.70330157]))


def test_Q_model_plotting(tmpdir):
    """
    Tests the plotting of the Q Model.
    """
    reset_matplotlib()

    tmpdir = str(tmpdir)

    weights = [2.50960201, 2.31899515, 0.19681762]
    relaxation_times = [1.73160984, 14.41562154, 16.70330157]
    Q_discrete.plot(weights, relaxation_times, f_min=1.0 / 100.0,
                    f_max=1.0 / 10.0)
    images_are_identical("discrete_Q_model", tmpdir)
