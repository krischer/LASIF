#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functionality for the LASIF test suite.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from collections import namedtuple
import copy
import inspect
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.testing.compare import compare_images as mpl_compare_images
import mock
import numpy as np
import obspy
import os
import pytest
import shutil
import sys

from lasif.components.project import Project
from lasif.scripts import lasif_cli

# Folder where all the images for comparison are stored.
IMAGES = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "baseline_images")
# Data path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))), "data")


@pytest.fixture
def dummy_project(tmpdir):
    """
    Fixture returning a communicator instance for a dummy project. The
    fixture has a couple of convenience method to modify the project to test
    different things.
    """
    # Create and empty project and get the communicator instance.
    project_path = os.path.join(str(tmpdir), "DummyProject")
    proj = Project(project_root_path=project_path, init_project="DummyProject")
    return proj, proj.comm


@pytest.fixture
def communicator(tmpdir):
    """
    Fixture returning the initialized example project. It will be a fresh copy
    every time that will be deleted after the test has finished so every test
    can mess with the contents of the folder.
    """
    # A new project will be created many times. ObsPy complains if objects that
    # already exists are created again.
    obspy.core.event.ResourceIdentifier\
        ._ResourceIdentifier__resource_id_weak_dict.clear()

    # Copy the example project
    example_project = os.path.join(DATA, "ExampleProject")
    project_path = os.path.join(str(tmpdir), "ExampleProject")
    shutil.copytree(example_project, project_path)

    # Init it. This will create the missing paths.
    return Project(project_path).comm


@pytest.fixture
def cli(communicator, request, capsys):
    """
    Fixture for being able to easily test the command line interface.

    Usage:
        stdout = cli.run("lasif info")
    """
    request.comm = communicator

    Output = namedtuple("Output", ["stdout", "stderr"])

    def run(command):
        old_dir = os.getcwd()
        old_argv = copy.deepcopy(sys.argv)
        try:
            # Change the path to the root path of the project.
            os.chdir(os.path.abspath(communicator.project.paths["root"]))
            components = command.split()
            if components[0] != "lasif":
                msg = "Invalid LASIF CLI command."
                raise Exception(msg)
            sys.argv = components
            # Greatly speed up some tests.
            with mock.patch("lasif.tools.Q_discrete.calculate_Q_model") as p:
                p.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                                  np.array([0.59496, 3.7119, 22.2171]))
                capsys.readouterr()
                try:
                    lasif_cli.main()
                except SystemExit:
                    pass
        except Exception as exc:
            raise exc
        finally:
            # Reset environment
            os.chdir(old_dir)
            sys.arv = old_argv
        return Output(*capsys.readouterr())

    request.run = run
    return request


def images_are_identical(image_name, temp_dir, dpi=None, tol=5):
    """
    Partially copied from ObsPy
    """
    image_name += os.path.extsep + "png"
    expected = os.path.join(IMAGES, image_name)
    actual = os.path.join(temp_dir, image_name)

    if dpi:
        plt.savefig(actual, dpi=dpi)
    else:
        plt.savefig(actual)
    plt.close()

    assert os.path.exists(expected)
    assert os.path.exists(actual)

    # Use a reasonably high tolerance to get around difference with different
    # freetype and possibly agg versions. matplotlib uses a tolerance of 13.
    result = mpl_compare_images(expected, actual, tol=tol, in_decorator=True)
    if result is not None:
        print result
    assert result is None


def reset_matplotlib():
    # Set all default values.
    mpl.rcdefaults()
    # These settings must be hardcoded for running the comparision tests and
    # are not necessarily the default values.
    mpl.rcParams['font.family'] = 'Bitstream Vera Sans'
    mpl.rcParams['text.hinting'] = False
    # Not available for all matplotlib versions.
    try:
        mpl.rcParams['text.hinting_factor'] = 8
    except KeyError:
        pass
    import locale
    locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))
