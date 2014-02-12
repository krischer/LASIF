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
from lxml import etree
import os

from lasif.iteration_xml import Iteration


# Most generic way to get the actual data directory.
data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


def test_reading_iteration_xml():
    """
    Tests the reading of an IterationXML file to the internal representation.
    """
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

    # Test some settings. The rest should work just fine as this is parsed
    # by useing a recursive dictionary.
    solver_params = iteration.solver_settings
    assert solver_params["solver"] == "SES3D 4.0"
    solver_settings = solver_params["solver_settings"]
    assert solver_settings["simulation_parameters"] == {
        "number_of_time_steps": 4000,
        "time_increment": 0.13,
        "is_dissipative": True
    }

    # Assert the events and stations up to a certain extend.
    assert len(iteration.events) == 2

    assert "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15" in iteration.events
    assert "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11" in iteration.events

    event_1 = iteration.events["GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"]
    assert event_1["event_weight"] == 1.0
    assert event_1["time_correction_in_s"] == 0.0
    stations = event_1["stations"]
    assert len(stations) == 8

    assert sorted(stations.keys()) == \
        sorted(["HL.ARG", "IU.ANTO", "GE.ISP", "HL.RDO", "HT.SIGR",
                "GE.SANT", "HT.ALN", "HL.SANT"])

    assert set([_i["station_weight"] for _i in stations.values()]) == \
        set([1.0])
    assert set([_i["time_correction_in_s"] for _i in stations.values()]) == \
        set([0.0])


def test_reading_and_writing(tmpdir):
    """
    Tests that reading and writing a file via IterationXML does not alter it.
    This effectively tests the Iteration XML writing.
    """
    filename = os.path.join(data_dir, "iteration_example.xml")
    new_filename = os.path.join(str(tmpdir), "iteration.xml")

    iteration = Iteration(filename)
    iteration.write(new_filename)

    # Compare the lxml etree's to avoid any difference in formatting and
    # what not.
    tree_old = etree.parse(filename)
    tree_new = etree.parse(new_filename)
    assert tree_old == tree_new
