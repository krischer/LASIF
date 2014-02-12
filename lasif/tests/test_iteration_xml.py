#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test cases for the iteration xml file handling.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import os

from lasif.iteration_xml import Iteration


# Most generic way to get the actual data directory.
data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


def test_reading_iteration_xml():
    iteration = Iteration(os.path.join(data_dir, "iteration_example.xml"))
    assert iteration.iteration_name == "IterationName"
    assert iteration.description == "Some description"
    assert iteration.comments == ["Comment 1", "Comment 2", "Comment 3"]
    assert iteration.data_preprocessing == {
        "highpass_period": 100.0,
        "lowpass_period": 8.0
    }
    assert iteration.source_time_function == "Filtered Heaviside"
    assert iteration.rejection_criteria == {
        "minimum_trace_length_in_s": 500.0,
        "signal_to_noise": {
            "test_interval_from_origin_in_s": 100.0,
            "max_amplitude_ratio": 100.0
        }
    }
