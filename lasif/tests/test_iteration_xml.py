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
import copy
import inspect
from lxml import etree
import os

from lasif.iteration_xml import Iteration


# Most generic way to get the actual data directory.
data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


def __stf_fct_dummy(*args, **kwargs):
    """
    Dummy stf function. In a proper LASIF project, the user-supplied project
    specific function will be passed.
    """
    pass


def test_reading_iteration_xml():
    """
    Tests the reading of an IterationXML file to the internal representation.
    """
    iteration = Iteration(os.path.join(data_dir, "iteration_example.xml"),
                          stf_fct=__stf_fct_dummy)
    # The name is always dependent on the filename
    assert iteration.iteration_name == "iteration_example"
    assert iteration.description == "Some description"
    assert iteration.comments == ["Comment 1", "Comment 2", "Comment 3"]
    assert iteration.data_preprocessing == {
        "highpass_period": 100.0,
        "lowpass_period": 8.0
    }

    # Test some settings. The rest should work just fine as this is parsed
    # by useing a recursive dictionary.
    solver_params = iteration.solver_settings
    assert solver_params["solver"] == "SES3D 4.1"
    solver_settings = solver_params["solver_settings"]
    assert solver_settings["simulation_parameters"] == {
        "number_of_time_steps": 4000,
        "time_increment": 0.13,
        "is_dissipative": True
    }

    # Small type check.
    assert isinstance(
        solver_settings["simulation_parameters"]["number_of_time_steps"], int)

    # Assert the events and stations up to a certain extend.
    assert len(iteration.events) == 2

    assert "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15" in iteration.events
    assert "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11" in iteration.events

    event_1 = iteration.events["GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"]
    assert event_1["event_weight"] == 1.0
    stations = event_1["stations"]
    assert len(stations) == 8

    assert sorted(stations.keys()) == \
        sorted(["HL.ARG", "IU.ANTO", "GE.ISP", "HL.RDO", "HT.SIGR",
                "GE.SANT", "HT.ALN", "HL.SANT"])

    assert set([_i["station_weight"] for _i in stations.values()]) == \
        set([1.0])

    # Test reading of comments for single events and stations.
    event_with_comments = iteration.events[
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"]

    assert event_with_comments["comments"] == [
        "This is some event, I tell you.",
        "Another comment just to test that multiple ones work."]

    station_with_comments = event_with_comments["stations"]["GE.APE"]
    assert station_with_comments["comments"] == [
        "Stations can also have comments!",
        "Who would have known?"]


def test_reading_and_writing(tmpdir):
    """
    Tests that reading and writing a file via IterationXML does not alter it.
    This effectively tests the Iteration XML writing.
    """
    filename = os.path.join(data_dir, "iteration_example.xml")
    new_filename = os.path.join(str(tmpdir), "iteration.xml")

    iteration = Iteration(filename, stf_fct=__stf_fct_dummy)
    iteration.write(new_filename)

    # Compare the lxml etree's to avoid any difference in formatting and
    # what not.
    tree_old = etree.tounicode(etree.parse(filename), pretty_print=True)
    tree_new = etree.tounicode(etree.parse(new_filename), pretty_print=True)

    # pytest takes care of meaningful string differences.
    assert tree_old == tree_new


def test_iteration_equality():
    """
    Tests equality/inequality for iteration xml files.
    """
    filename = os.path.join(data_dir, "iteration_example.xml")

    iteration = Iteration(filename, stf_fct=__stf_fct_dummy)
    other_iteration = copy.deepcopy(iteration)

    assert iteration == other_iteration
    assert not iteration != other_iteration

    iteration.iteration_name = "blub"
    assert iteration != other_iteration
    assert not iteration == other_iteration


def test_reading_writing_with_empty_description(tmpdir):
    """
    Tests reading and writing with an empty description.
    """
    filename = os.path.join(data_dir, "iteration_example.xml")
    new_filename = os.path.join(str(tmpdir), "iteration.xml")

    iteration = Iteration(filename, stf_fct=__stf_fct_dummy)
    iteration.description = None

    # Write and read again.
    iteration.write(new_filename)
    reread_iteration = Iteration(new_filename, stf_fct=__stf_fct_dummy)

    # Change the name as it is always dependent on the filename.
    reread_iteration.iteration_name = iteration.iteration_name
    assert iteration == reread_iteration
