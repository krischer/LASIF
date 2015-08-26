#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import mock
import numpy as np
import os
import pytest
import shutil

from lasif.components.project import Project

from ..testing_helpers import images_are_identical, reset_matplotlib


def setup_function(function):
    """
    Reset matplotlib.
    """
    reset_matplotlib()


@pytest.fixture()
def comm(tmpdir):
    """
    Most visualizations need a valid project in any case, so use one for the
    tests.
    """
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "ExampleProject")
    tmpdir = str(tmpdir)
    shutil.copytree(proj_dir, os.path.join(tmpdir, "proj"))
    proj_dir = os.path.join(tmpdir, "proj")

    project = Project(project_root_path=proj_dir, init_project=False)

    return project.comm


def test_event_plotting(comm):
    """
    Tests the plotting of all events.

    The commands supports three types of plots: Beachballs on a map and depth
    and time distribution histograms.
    """
    comm.visualizations.plot_events(plot_type="map")
    images_are_identical("two_events_plot_map", comm.project.paths["root"])

    comm.visualizations.plot_events(plot_type="depth")
    images_are_identical("two_events_plot_depth", comm.project.paths["root"])

    comm.visualizations.plot_events(plot_type="time")
    images_are_identical("two_events_plot_time", comm.project.paths["root"])


def test_single_event_plot(comm):
    """
    Tests the plotting of a single event.
    """
    comm.visualizations.plot_event("GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    images_are_identical("single_event_plot", comm.project.paths["root"])


def test_simple_raydensity(comm):
    """
    Test plotting a simple raydensity map.
    """
    comm.visualizations.plot_raydensity(save_plot=False)
    # Use a low dpi to keep the test filesize in check.
    images_are_identical("simple_raydensity_plot", comm.project.paths["root"],
                         dpi=25)


def test_simple_raydensity_with_stations(comm):
    """
    Test plotting a simple raydensity map with stations.
    """
    comm.visualizations.plot_raydensity(save_plot=False, plot_stations=True)
    # Use a low dpi to keep the test filesize in check.
    images_are_identical("simple_raydensity_plot_with_stations",
                         comm.project.paths["root"], dpi=25)


@mock.patch("lasif.tools.Q_discrete.calculate_Q_model")
def test_Q_model_plotting(patch, comm):
    """
    Tests the Q model plotting with mocking. The actual plotting is tested
    at the Q model tests.
    """
    # Speed up this test.
    patch.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

    comm.iterations.create_new_iteration(
        iteration_name="1", solver_name="ses3d_4_1",
        events_dict=comm.query.get_stations_for_all_events(),
        min_period=11, max_period=111)

    with mock.patch("lasif.tools.Q_discrete.plot") as patch:
        comm.iterations.plot_Q_model("1")
        assert patch.call_count == 1
        kwargs = patch.call_args[1]
    assert round(kwargs["f_min"] - 1.0 / 111.0, 5) == 0
    assert round(kwargs["f_max"] - 1.0 / 11.0, 5) == 0
