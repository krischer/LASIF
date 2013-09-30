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


def test_weights_and_relaxation_times():
    """
    Regression test for the weights and relaxation times.

    Due to them being the result of a non-linear and non-unique inversion, the
    results are different from run to run thus they are hard to test. At least
    this test somehow asserts that the function does not fail.
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
    assert len(weights) == 3
    assert len(relaxation_times) == 3
    # The big weights are usually similar enough.
    s_weights = sorted(weights)[1:]
    ref = [2.46112185, 2.51074556]
    assert abs(s_weights[0] - ref[0]) < 1.0
    assert abs(s_weights[1] - ref[1]) < 1.0
    # The relaxation times can differ quite a bit from realization to
    # realization. Thus they are not really tested.
    assert sum(relaxation_times) > 10.0
