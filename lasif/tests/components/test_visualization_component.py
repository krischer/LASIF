#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import pathlib

import numpy as np
import os
import pytest
import shutil
from unittest import mock

from lasif.components.project import Project
from ..testing_helpers import images_are_identical, reset_matplotlib


def setup_function(function):
    """
    Reset matplotlib.
    """
    reset_matplotlib()

@pytest.fixture()
def comm(tmpdir):
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "ExampleProject")
    tmpdir = str(tmpdir)
    shutil.copytree(proj_dir, os.path.join(tmpdir, "proj"))
    proj_dir = os.path.join(tmpdir, "proj")

    folder_path = pathlib.Path(proj_dir).absolute()
    project = Project(project_root_path=folder_path, init_project=False)
    os.chdir(os.path.abspath(folder_path))
    #assert False, (folder_path)

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
    images_are_identical("two_events_plot_depth",
                         comm.project.paths["root"],
                         tol=30)

    comm.visualizations.plot_events(plot_type="time")
    images_are_identical("two_events_plot_time", comm.project.paths["root"],
                         tol=30)

#
# def test_single_event_plot(comm):
#     """
#     Tests the plotting of a single event.
#     """
#     comm.visualizations.plot_event("GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
#     images_are_identical("single_event_plot", comm.project.paths["root"])
#
#
# def test_simple_raydensity(comm):
#     """
#     Test plotting a simple raydensity map.
#     """
#     comm.visualizations.plot_raydensity(save_plot=False)
#     # Use a low dpi to keep the test filesize in check.
#     images_are_identical("simple_raydensity_plot",
#                          comm.project.paths["root"], dpi=25)
#
#
# def test_simple_raydensity_with_stations(comm):
#     """
#     Test plotting a simple raydensity map with stations.
#     """
#     comm.visualizations.plot_raydensity(save_plot=False, plot_stations=True)
#     # Use a low dpi to keep the test filesize in check.
#     images_are_identical("simple_raydensity_plot_with_stations",
#                          comm.project.paths["root"], dpi=25)
#
