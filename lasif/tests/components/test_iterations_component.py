#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import pathlib

import numpy as np
import pytest
from unittest import mock

import shutil
import os

from lasif.components.iterations import IterationsComponent
from lasif.components.communicator import Communicator
from lasif.components.project import Project


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
    folder_path = pathlib.Path(proj_dir).absolute()
    project = Project(project_root_path=folder_path, init_project=False)

    return project.comm
