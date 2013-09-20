#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test cases for the CLI interface.

Many of these test are very similar to the project tests. But this is what the
CLI interface is supposed to provide: an easy way to interface with the project
class.

Furthermore many tests are simple mock tests only asserting that the proper
methods are called.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import matplotlib as mpl
mpl.use("agg")

import mock

from lasif.scripts import lasif_cli

from lasif.tests.testing_helpers import project, cli  # NOQA
from lasif.tests.testing_helpers import reset_matplotlib

# Get a list of all available commands.
CMD_LIST = [key.replace("lasif_", "")
            for (key, value) in lasif_cli.__dict__.iteritems()
            if (key.startswith("lasif_") and callable(value))]


def setup_function(function):
    """
    Make sure matplotlib behaves the same on every machine.
    """
    reset_matplotlib()


def test_test_sanity():
    """
    Quick test to test the tests...
    """
    assert len(CMD_LIST) >= 10
    assert "info" in CMD_LIST
    assert "init_project" in CMD_LIST


def test_invocation_without_parameters(cli):
    """
    Tests the invocation without any parameters.
    """
    default_output = cli.run("lasif")
    # Should be the same as if invoced with --help.
    assert default_output == cli.run("lasif --help")
    # It should furthermore contain a list of all commands.
    for cmd in CMD_LIST:
        assert cmd in default_output.stdout


def test_help_messages(cli):
    """
    Tests the help messages.
    """
    for cmd in CMD_LIST:
        # Both invocations should work
        assert cli.run("lasif %s --help" % cmd) == \
            cli.run("lasif help %s" % cmd)
        # Some things should always be shown. This also more or less tests that
        # the argparse parser is used everywhere.
        help_string = cli.run("lasif %s --help" % cmd).stdout
        assert help_string.startswith("usage: lasif %s" % cmd)
        assert "show this help message and exit" in help_string
        assert "optional arguments:" in help_string


def test_project_init(cli):
    """
    Tests the project initialization with the CLI interface.
    """
    # Invocation without a folder path fails.
    log = cli.run("lasif init_project")
    assert "error: too few arguments" in log.stderr


def test_plotting_functions(cli):
    """
    Tests if the correct plotting functions are called.
    """
    with mock.patch("lasif.project.Project.plot_domain") as patch:
        cli.run("lasif plot_domain")
        patch.assert_called_once_with()

    with mock.patch("lasif.project.Project.plot_event") as patch:
        cli.run("lasif plot_event EVENT_NAME")
        patch.assert_called_once_with("EVENT_NAME")

    # Test the different variations of the plot_events function.
    with mock.patch("lasif.project.Project.plot_events") as patch:
        cli.run("lasif plot_events")
        patch.assert_called_once_with("map")

    with mock.patch("lasif.project.Project.plot_events") as patch:
        cli.run("lasif plot_events --type=map")
        patch.assert_called_once_with("map")

    with mock.patch("lasif.project.Project.plot_events") as patch:
        cli.run("lasif plot_events --type=time")
        patch.assert_called_once_with("time")

    with mock.patch("lasif.project.Project.plot_events") as patch:
        cli.run("lasif plot_events --type=depth")
        patch.assert_called_once_with("depth")

    # Misc plotting functionality.
    with mock.patch("lasif.project.Project.plot_raydensity") as patch:
        cli.run("lasif plot_raydensity")
        patch.assert_called_once_with()
