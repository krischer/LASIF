#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for some utility functions.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import numpy as np
import os
import pytest
import pathlib
import shutil

from lasif import utils
from lasif.components.project import Project

data_dir = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


@pytest.fixture()
def comm(tmpdir):
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "tests", "data",
        "ExampleProject")
    tmpdir = str(tmpdir)
    shutil.copytree(proj_dir, os.path.join(tmpdir, "proj"))
    proj_dir = os.path.join(tmpdir, "proj")

    folder_path = pathlib.Path(proj_dir).absolute()
    project = Project(project_root_path=folder_path, init_project=False)
    os.chdir(os.path.abspath(folder_path))

    return project.comm


def test_greatcircle_points_generator():
    """
    Tests the greatcircle discrete point generator.
    """
    points = list(utils.greatcircle_points(
        utils.Point(0, 0), utils.Point(0, 90), max_npts=90))
    assert len(points) == 90
    assert [_i.lat for _i in points] == 90 * [0.0]
    np.testing.assert_array_almost_equal([_i.lng for _i in points],
                                         np.linspace(0, 90, 90))

    points = list(utils.greatcircle_points(
        utils.Point(0, 0), utils.Point(0, 90), max_npts=110))
    assert len(points) == 110

    # from obspy import read_events
    # filename = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    #     inspect.getfile(inspect.currentframe())))), "tests", "data",
    #     "ExampleProject", "DATA", "EARTHQUAKES")
    # read_events(filename, format="ndk")
