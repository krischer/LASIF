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

from lasif.tools import Q_discrete
from lasif.tests.testing_helpers import images_are_identical, \
    reset_matplotlib


WEIGHTS = np.array([1.6264684983257656, 1.0142952434286228,
                    1.5007527644957979])
RELAXATION_TIMES = np.array([0.68991741458188449, 4.1538611409236301,
                             23.537531778655516])


def test_weights_and_relaxation_times():
    """
    Regression test for the weights and relaxation times.
    """
    # Set the seed to get reproducible results.
    np.random.seed(12345)

    # These are the D_p and tau_p, respectively.
    weights, relaxation_times, = Q_discrete.calculate_Q_model(
        N=3,
        f_min=1.0 / 100.0,
        f_max=1.0 / 10.0,
        iterations=10000,
        initial_temperature=0.1,
        cooling_factor=0.9998)

    np.testing.assert_array_almost_equal(weights, WEIGHTS)

    np.testing.assert_array_almost_equal(relaxation_times, RELAXATION_TIMES)


def test_Q_model_plotting(tmpdir):
    """
    Tests the plotting of the Q Model.
    """
    reset_matplotlib()

    tmpdir = str(tmpdir)

    Q_discrete.plot(WEIGHTS, RELAXATION_TIMES, f_min=1.0 / 100.0,
                    f_max=1.0 / 10.0)
    images_are_identical("discrete_Q_model", tmpdir)
