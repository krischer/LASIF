#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test cases for the CLI interface.

Many of these test are very similar to the project tests. But this is what the
CLI interface is supposed to provide: an easy way to interface with the project
class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import matplotlib as mpl
mpl.use("agg")

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
    for cmd in CMD_LIST:
        assert cmd in default_output
