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

import numpy as np
import mock
import os

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


def test_command_tolerance(cli):
    """
    Tests that upper and lowercase subcommands are not distinguished.
    """
    with mock.patch("lasif.scripts.lasif_cli.lasif_info") as patch:
        cli.run("lasif info")
        patch.assert_called_once()

    with mock.patch("lasif.scripts.lasif_cli.lasif_info") as patch:
        cli.run("lasif INFO")
        patch.assert_called_once()

    with mock.patch("lasif.scripts.lasif_cli.lasif_info") as patch:
        cli.run("lasif InFo")
        patch.assert_called_once()


def test_unknown_command(cli):
    """
    Tests the message when an unknown command is called.
    """
    out = cli.run("lasif asdflkjaskldfj")
    assert out.stdout == ""
    assert out.stderr == ("lasif: 'asdflkjaskldfj' is not a LASIF command. "
                          "See 'lasif --help'.")


def test_fuzzy_command_matching(cli):
    """
    If the user enters a slightly wrong subcommand, the user should be notified
    of alternatives.
    """
    out = cli.run("lasif infi")
    assert out.stdout == ""
    assert out.stderr == (
        "lasif: 'infi' is not a LASIF command. See 'lasif --help'.\n\n"
        "Did you mean this?\n"
        "\tinfo")

    out = cli.run("lasif plot_eventos")
    assert out.stdout == ""
    assert out.stderr == (
        "lasif: 'plot_eventos' is not a LASIF command. See 'lasif --help'.\n\n"
        "Did you mean one of these?\n"
        "\tplot_event\n"
        "\tplot_events\n")


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


def test_download_utitlies(cli):
    """
    Testing the invocation of the downloaders.
    """
    # SPUD interface downloader.
    with mock.patch("lasif.scripts.iris2quakeml.iris2quakeml") as patch:
        cli.run("lasif add_spud_event https://test.org")
        patch.assert_called_once_with(
            "https://test.org", cli.project.paths["events"])

    # Test the waveform downloader invocation.
    with mock.patch("lasif.download_helpers.downloader.download_waveforms") \
            as download_patch, \
            mock.patch("lasif.project.Project.is_event_in_project") \
            as is_event_patch:
        cli.run("lasif download_waveforms "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
        is_event_patch.return_value = True
        is_event_patch.assert_called_with(
            "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
        download_patch.assert_called_once()

    # Test the station downloader invocation.
    with mock.patch("lasif.download_helpers.downloader.download_stations") \
            as download_patch, \
            mock.patch("lasif.project.Project.is_event_in_project") \
            as is_event_patch:
        cli.run("lasif download_stations "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
        is_event_patch.return_value = True
        is_event_patch.assert_called_with(
            "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
        download_patch.assert_called_once()


def test_lasif_info(cli):
    """
    Tests the 'lasif info' command.
    """
    out = cli.run("lasif info").stdout
    assert "\"ExampleProject\"" in out
    assert "Toy Project used in the Test Suite" in out
    assert "2 events" in out
    assert "4 station files" in out
    assert "4 raw waveform files" in out
    assert "0 processed waveform files" in out
    assert "6 synthetic waveform files" in out


def test_various_list_functions(cli):
    """
    Tests all the "lasif list_" functions.
    """
    events = cli.run("lasif list_events").stdout
    assert "2 events" in events
    assert "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11" in events
    assert "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15" in events

    iterations = cli.run("lasif list_iterations").stdout
    assert "0 iterations" in iterations
    with open(os.path.join(cli.project.paths["iterations"], "1.xml"),
              "wt") as fh:
        fh.write("<>")
    iterations = cli.run("lasif list_iterations").stdout
    assert "1 iteration" in iterations
    with open(os.path.join(cli.project.paths["iterations"], "temp.xml"),
              "wt") as fh:
        fh.write("<>")
    iterations = cli.run("lasif list_iterations").stdout
    assert "2 iteration" in iterations

    models = cli.run("lasif list_models").stdout
    assert "0 models" in models
    os.makedirs(os.path.join(cli.project.paths["models"], "BLUB"))
    models = cli.run("lasif list_models").stdout
    assert "1 model" in models


def test_iteration_creation_and_stf_plotting(cli):
    """
    Tests the generation of an iteration and the supsequent STF plotting.
    """
    cli.run("lasif create_new_iteration 1 SES3D_4_0")
    assert "1" in cli.project.get_iteration_dict().keys()

    with mock.patch("lasif.visualization.plot_tf") as patch:
        cli.run("lasif plot_stf 1")
        patch.assert_called_once()
        data, delta = patch.call_args[0]
        np.testing.assert_array_equal(
            data,
            cli.project._get_iteration("1").get_source_time_function()["data"])
        assert delta == 0.75


def test_lasif_event_info(cli):
    """
    Tests the event info function.
    """
    event_1 = cli.run("lasif event_info "
                      "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11").stdout
    event_2 = cli.run("lasif event_info "
                      "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15").stdout

    assert "5.1 Mw" in event_1
    assert "TURKEY" in event_1
    assert "38.820" in event_1
    assert "available at 4 stations" in event_1

    assert "5.9 Mw" in event_2
    assert "TURKEY" in event_2
    assert "39.150" in event_2
    assert "available at 0 stations" in event_2


def test_input_file_generatioN(cli):
    """
    Mock test to see if the input file generation routine is called. The
    routine is tested partially by the event tests and more by the input file
    generation module.
    """
    # No solver specified.
    with mock.patch("lasif.project.Project.generate_input_files") as patch:
        cli.run("lasif generate_input_files 1 "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
        patch.assert_called_once_with(
            "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
            "normal simulation")

    # Normal simulation
    with mock.patch("lasif.project.Project.generate_input_files") as patch:
        cli.run("lasif generate_input_files 1 "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11 "
                "--simulation_type=normal_simulation")
        patch.assert_called_once_with(
            "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
            "normal simulation")

    # Adjoint forward.
    with mock.patch("lasif.project.Project.generate_input_files") as patch:
        cli.run("lasif generate_input_files 1 "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11 "
                "--simulation_type=adjoint_forward")
        patch.assert_called_once_with(
            "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
            "adjoint forward")

    # Adjoint reverse.
    with mock.patch("lasif.project.Project.generate_input_files") as patch:
        cli.run("lasif generate_input_files 1 "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11 "
                "--simulation_type=adjoint_reverse")
        patch.assert_called_once_with(
            "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
            "adjoint reverse")


def test_finalize_adjoint_sources(cli):
    """
    Simple mock test.
    """
    with mock.patch("lasif.project.Project.finalize_adjoint_sources") as p:
        cli.run("lasif generate_input_files 1 "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
        p.assert_calles_once_with(
            "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")


def test_preprocessing_and_launch_misfit_gui(cli):
    """
    Tests the proprocessing and the launching of the misfit gui. Both are done
    together because the former is required by the later and takes a rather
    long time.
    """
    cli.run("lasif create_new_iteration 1 SES3D_4_0")

    processing_tag = cli.project._get_iteration("1").get_processing_tag()
    preprocessing_data = os.path.join(
        cli.project.paths["data"], "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
        processing_tag)
    assert not os.path.exists(preprocessing_data)
    cli.run("lasif preprocess_data 1")
    assert os.path.exists(preprocessing_data)
    assert len(os.listdir(preprocessing_data)) == 4

    with mock.patch("lasif.misfit_gui.MisfitGUI") as patch:
        cli.run("lasif launch_misfit_gui 1 "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
        patch.assert_called_once()
        ev, it, proj, wm, ad_m = patch.call_args[0]
        assert ev == cli.project.get_event(
            "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
        assert proj.paths["root"] == cli.project.paths["root"]
        assert it.__class__.__name__ == "TwoWayIter"
        assert wm.__class__.__name__ == "MisfitWindowManager"
        assert ad_m.__class__.__name__ == "AdjointSourceManager"


def test_iteration_info(cli):
    """
    Tests the 'lasif iteration_info' command.
    """
    cli.run("lasif create_new_iteration 1 SES3D_4_0")

    out = cli.run("lasif iteration_info 1").stdout
    assert "LASIF Iteration" in out
    assert "Name: 1" in out
    assert "Solver: SES3D 4.0" in out


def test_remove_empty_coordinate_entries(cli):
    """
    Simple mock test.
    """
    with mock.patch("lasif.tools.inventory_db.reset_coordinate_less_stations")\
            as patch:
        cli.run("lasif remove_empty_coordinate_entires")
        assert patch.assert_run_once_with(cli.project.paths["inv_db_file"])


def test_validate_data(cli):
    """
    Simple mock test.
    """
    with mock.patch("lasif.project.Project.validate_data") as patch:
        cli.run("lasif validate_data")
        patch.assert_called_once_with(full_check=False)

    with mock.patch("lasif.project.Project.validate_data") as patch:
        cli.run("lasif validate_data --full")
        patch.assert_called_once_with(full_check=True)


def test_open_tutorial(cli):
    """
    Simple mock test.
    """
    with mock.patch("webbrowser.open") as patch:
        cli.run("lasif tutorial")
        patch.assert_called_once_with("http://krischer.github.io/LASIF/")


def test_iteration_status_command(cli):
    """
    The iteration status command returns the current state of any iteration. It
    returns the number of already preprocessed data files, how many synthetics
    are available, the windows and adjoint sources.
    """
    cli.run("lasif create_new_iteration 1 SES3D_4_0")
    out = cli.run("lasif iteration_status 1").stdout
    assert out == (
        "Iteration Name: 1\n"
        "\tAll necessary files available.\n"
        "\t4 out of 4 files still require preprocessing.\n")

    cli.run("lasif preprocess_data 1")
    out = cli.run("lasif iteration_status 1").stdout
    assert out == (
        "Iteration Name: 1\n"
        "\tAll necessary files available.\n"
        "\tAll files are preprocessed.\n")
