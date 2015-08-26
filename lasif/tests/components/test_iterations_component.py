#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import mock
import numpy as np
import pytest

from lasif.components.iterations import IterationsComponent
from lasif.components.communicator import Communicator


@pytest.fixture()
def comm(tmpdir):
    """
    Fixture for the iterations tests. This already requires parts of the
    iterations component to be working but that is needed for the tests in
    any case.
    """
    tmpdir = str(tmpdir)
    comm = Communicator()

    # Mock the project specific source time function.
    comm.project = mock.MagicMock()
    comm.project.get_project_function = lambda a: None

    IterationsComponent(
        iterations_folder=tmpdir,
        communicator=comm,
        component_name="iterations")

    # Greatly speed up this fixture.
    with mock.patch("lasif.tools.Q_discrete.calculate_Q_model") as p:
        p.return_value = (np.array([1.6341, 1.0513, 1.5257]),
                          np.array([0.59496, 3.7119, 22.2171]))

        # Create two iterations.
        comm.iterations.create_new_iteration(
            "1", "ses3d_4_1",
            {"EVENT_1": ["AA.BB", "CC.DD"], "EVENT_2": ["EE.FF"]}, 10.0, 20.0,
            quiet=True, create_folders=False)
        comm.iterations.create_new_iteration(
            "2", "ses3d_4_1",
            {"EVENT_1": ["AA.BB", "CC.DD"], "EVENT_2": ["EE.FF"]}, 10.0, 20.0,
            quiet=True, create_folders=False)

    it = comm.iterations.get("1")
    it.comments = ["Some", "random comments"]
    comm.iterations.save_iteration(it)
    return comm
