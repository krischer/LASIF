# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# Test cases for the CLI interface.
#
# Many of these test are very similar to the project tests.
# But this is what the
# CLI interface is supposed to provide:
#  an easy way to interface with the project's components.
#
# Furthermore many tests are simple mock tests only asserting that the proper
# methods are called.
#
# In many cases ``patch.assert_called_once_with()`` and
# ``assert patch.call_count == 1`` is used. That is because it is really easy
# to mistype the method and then mock just ignores it resulting in nothing
# being actually tested. So its just an additional design decision.
#
# :copyright:
#     Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
# :license:
#     GNU General Public License, Version 3
#     (http://www.gnu.org/copyleft/gpl.html)
# """
import matplotlib as mpl
mpl.use("agg")

# import numpy as np
import os
import shutil
import numpy as np
from unittest import mock

import lasif
from lasif.scripts import lasif_cli
from lasif.tests.testing_helpers import reset_matplotlib
from lasif.tests.testing_helpers import communicator, cli  # NOQA


# Get a list of all available commands.
CMD_LIST = [key.replace("lasif_", "")
            for (key, value) in lasif_cli.__dict__.items()
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
        # Some things should always be shown.
        # This also more or less tests that
        # the argparse parser is used everywhere.
        help_string = cli.run("lasif %s --help" % cmd).stdout
        # print(help_string)
        assert help_string.startswith("usage: lasif %s" % cmd)
        assert "show this help message and exit" in help_string
        assert "optional arguments:" in help_string


def test_command_tolerance(cli):
    """
    Tests that upper and lowercase subcommands are not distinguished.
    """
    assert cli.run("lasif info").stdout == cli.run("lasif INFO").stdout
    assert cli.run("lasif info").stdout == cli.run("lasif InFo").stdout


def test_unknown_command(cli):
    """
    Tests the message when an unknown command is called.
    """
    out = cli.run("lasif asdflkjaskldfj")
    assert out.stdout == ""
    assert out.stderr == ("lasif: 'asdflkjaskldfj' is not a LASIF command. "
                          "See 'lasif --help'.\n")


def test_fuzzy_command_matching(cli):
    """
    If the user enters a slightly wrong subcommand,
    the user should be notified of alternatives.
    """
    out = cli.run("lasif infi")
    assert out.stdout == ""
    assert out.stderr == (
        "lasif: 'infi' is not a LASIF command. See 'lasif --help'.\n\n"
        "Did you mean this?\n"
        "\tinfo\n")

    out = cli.run("lasif plot_eventos")
    assert out.stdout == ""
    assert out.stderr == (
        "lasif: 'plot_eventos' is not a LASIF command. See 'lasif --help'.\n\n"
        "Did you mean one of these?\n"
        "    list_events\n"
        "    plot_event\n"
        "    plot_events\n"
        "    plot_windows\n")


def test_cli_parsing_corner_cases(cli):
    """
    Tests any funky corner cases related to the command line parsing.
    """
    out = cli.run("lasif help --help")
    assert out.stdout == ""
    assert out.stderr == "lasif: Invalid command. See 'lasif --help'.\n"


def test_project_init_without_arguments(cli):
    """
    Tests the project initialization with the CLI interface without passed
    arguments.
    """
    # Invocation without a folder path fails.
    log = cli.run("lasif init_project")
    assert "error: the following arguments are required: folder_path" \
           in log.stderr
    log2 = cli.run("lasif plot_event")
    assert "error: the following arguments are required: event_name" \
           in log2.stderr


def test_project_init(cli):
    """
    Tests the project initialization.
    """
    # Delete all contents of directory to be able to start with a clean one.
    root_path = cli.comm.project.paths["root"]
    for filename in os.listdir(root_path):
        file_path = os.path.join(root_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            shutil.rmtree(file_path)

    # Initialize project.
    out = cli.run("lasif init_project TestDummy")
    assert out.stderr == ""
    assert "Initialized project in" in out.stdout


def test_plotting_functions(cli):
    """
    Tests if the correct plotting functions are called.
    """
    vs = "lasif.components.visualizations.VisualizationsComponent."
    with mock.patch(vs + "plot_domain") as patch:
        cli.run("lasif plot_domain")
    assert patch.call_count == 1

    with mock.patch(vs + "plot_event") as patch:
        cli.run("lasif plot_event event_name")
    patch.assert_called_once_with("event_name", None)
    assert patch.call_count == 1

    with mock.patch(vs + "plot_event") as patch:
        cli.run("lasif plot_event event_name --weight_set_name A")
    patch.assert_called_once_with("event_name", "A")
    assert patch.call_count == 1

    # Test the different variations of the plot_events function.
    with mock.patch(vs + "plot_events") as patch:
        cli.run("lasif plot_events")
    patch.assert_called_once_with("map")
    assert patch.call_count == 1

    with mock.patch(vs + "plot_events") as patch:
        cli.run("lasif plot_events --type=map")
    patch.assert_called_once_with("map")
    assert patch.call_count == 1

    with mock.patch(vs + "plot_events") as patch:
        cli.run("lasif plot_events --type=time")
    patch.assert_called_once_with("time")
    assert patch.call_count == 1

    with mock.patch(vs + "plot_events") as patch:
        cli.run("lasif plot_events --type=depth")
    patch.assert_called_once_with("depth")
    assert patch.call_count == 1

    # Misc plotting functionality.
    with mock.patch(vs + "plot_raydensity") as patch:
        cli.run("lasif plot_raydensity")
    patch.assert_called_once_with(plot_stations=False)
    assert patch.call_count == 1

    with mock.patch(vs + "plot_raydensity") as patch:
        cli.run("lasif plot_raydensity --plot_stations")
    patch.assert_called_once_with(plot_stations=True)
    assert patch.call_count == 1

    # Lacking the testing of plot_section. Need processed data.


def test_download_utitlies(cli):
    """
    Testing the invocation of the downloaders.
    """
    # SPUD interface downloader.
    with mock.patch("lasif.scripts.iris2quakeml.iris2quakeml") as patch:
        cli.run("lasif add_spud_event https://test.org")
    patch.assert_called_once_with(
        "https://test.org", cli.comm.project.paths["eq_data"])
    assert patch.call_count == 1

    # Test the download data invocation.
    with mock.patch("lasif.components.downloads.DownloadsComponent"
                    ".download_data") \
            as download_patch:
        out = cli.run("lasif download_data "
                      "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10")
    assert out.stderr == ""
    download_patch.assert_called_once_with(
        "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10", providers=None)
    assert download_patch.call_count == 1

    # Test setting the providers.
    with mock.patch("lasif.components.downloads.DownloadsComponent"
                    ".download_data") \
            as download_patch:
        out = cli.run("lasif download_data "
                      "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10 "
                      "--providers IRIS ORFEUS")
    assert out.stderr == ""
    download_patch.assert_called_once_with(
        "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10",
        providers=["IRIS", "ORFEUS"])
    assert download_patch.call_count == 1


def test_lasif_info(cli):
    """
    Tests the 'lasif info' command.
    """
    out = cli.run("lasif info").stdout
    assert "\"ExampleProject\"" in out
    assert "Toy Project used in the Test Suite" in out
    assert "2 events" in out
#     assert "4 station files" in out
#     assert "6 raw waveform files" in out
#     assert "0 processed waveform files" in out
#     assert "6 synthetic waveform files" in out


def test_various_list_functions(cli):
    """
    Tests all the "lasif list_" functions.
    """
    events = cli.run("lasif list_events").stdout
    assert "2 events" in events
    assert "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11" in events
    assert "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15" in events

    # Also has a --list option.
    events = cli.run("lasif list_events --list").stdout
    assert events == (
        "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11\n"
        "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15\n")

    iterations = cli.run("lasif list_iterations").stdout
    assert "1 iteration in this project" in iterations

    cli.run("lasif set_up_iteration 2").stdout
    iterations = cli.run("lasif list_iterations").stdout
    assert "2 iterations" in iterations


def test_iteration_creation(cli):
    """
    Tests the generation of an iteration and removal of one
    """
    cli.run("lasif set_up_iteration 3")
    assert cli.comm.iterations.has_iteration("3")
    cli.run("lasif set_up_iteration 3 --remove_dirs")
    assert not cli.comm.iterations.has_iteration("3")


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
    assert "available at 4 stations" in event_2


def test_plot_stf(cli):
    """
    Tests the source time function plots.
    """
    with mock.patch("lasif.visualization.plot_tf") as patch:
        cli.run("lasif plot_stf")
    assert patch.call_count == 1

    data, delta = patch.call_args[0]
    stf_fct = cli.comm.project.get_project_function("source_time_function")
    stf_delta = cli.comm.project.solver_settings["time_increment"]
    stf_npts = cli.comm.project.solver_settings["number_of_time_steps"]
    stf_freqmin = 1.0 / cli.comm.project.processing_params["lowpass_period"]
    stf_freqmax = 1.0 / cli.comm.project.processing_params["highpass_period"]

    stf_data = stf_fct(npts=stf_npts, delta=stf_delta, freqmin=stf_freqmin,
                       freqmax=stf_freqmax)
    np.testing.assert_array_equal(data, stf_data)
    assert stf_delta == delta


def test_generate_input_files(cli):
    """
    Mock test for generate_all_input_files.
    """
    ac = "lasif.components.actions.ActionsComponent."
    event = "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10"
    # Test if it gives an error if there is no iteration available
    with mock.patch(ac + 'generate_input_files') as patch:
        out = cli.run("lasif generate_input_files 2 " + event + ' forward')
    assert "Could not find iteration: 2" in out.stderr
    assert patch.call_count == 0

    # Now set up an iteration and then there should be no errors
    cli.run("lasif set_up_iteration 1")
    with mock.patch(ac + 'generate_input_files') as patch:
        out = cli.run("lasif generate_input_files 1 " + event + ' forward')
    assert out.stderr == ""
    assert "Generating input files for event " in out.stdout
    assert event in out.stdout
    patch.assert_called_once_with("1", event, 'forward')
    assert patch.call_count == 1

    # No simulation type specified:
    with mock.patch(ac + "generate_input_files") as patch:
        out = cli.run("lasif generate_input_files 1 " + event)
    assert 'Please choose simulation_type from: [forward, step_length, ' \
           'adjoint]' in out.stderr
    assert patch.call_count == 0

    # Step length simulation
    with mock.patch(ac + 'generate_input_files') as patch:
        out = cli.run("lasif generate_input_files 1 " + event + ' step_length')
    assert out.stderr == ""
    patch.assert_called_once_with("1", event, "step_length")
    assert patch.call_count == 1

    # Adjoint simulation should get errors now since there are no files ready
    with mock.patch(ac + 'generate_input_files') as patch:
        out = cli.run("lasif generate_input_files 1 " + event + ' adjoint')
    assert not out.stderr == ""
    assert patch.call_count == 0

    """
    We could make a test where adjoint sources are calculated prior to
    running the adjoint simulation but for now this will have to do.
    we could also have the adjoint sources ready in the example project
    and use that to run the test.
    """


def test_calculate_all_adjoint_sources(cli):
    """
    Simple mock test.
    """
    with mock.patch("lasif.components.actions.ActionsComponent"
                    ".calculate_adjoint_sources") as p:
        out = cli.run("lasif calculate_adjoint_sources 1 B "
                      "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    assert "Window set B not known to LASIF" in out.stderr
    assert p.call_count == 0

    with mock.patch("lasif.components.actions.ActionsComponent"
                    ".calculate_adjoint_sources") as p:
        out = cli.run("lasif calculate_adjoint_sources")
    assert "error: the following arguments are required: iteration_name, " \
           "window_set_name, events" in out.stderr
    assert p.call_count == 0

    """
    This can be improved by having a window set in the example project.
    Otherwise it's not really feasible to test
    """


def test_finalize_adjoint_sources(cli):
    """
    Simple mock test.
    """
    cli.run("lasif set_up_iteration 1")
    with mock.patch("lasif.components.actions.ActionsComponent"
                    ".finalize_adjoint_sources") as p:
        out = cli.run("lasif generate_input_files 1 "
                      "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11 adjoint")
    assert out.stderr == ""
    p.assert_called_once_with(
        "1", "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11", None)
    assert p.call_count == 1


def test_launch_misfit_gui(cli):
    with mock.patch("lasif.misfit_gui.misfit_gui.launch") as patch:
        cli.run("lasif launch_misfit_gui")

    assert patch.call_count == 1


def test_preprocessing(cli):
    """
    Tests the processing.
    """
    event = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    processing_path = cli.comm.project.paths["preproc_eq_data"]
    event_path = os.path.join(processing_path, event)

    import shutil
    import glob
    for path in glob.glob(os.path.join(processing_path, "*")):
        shutil.rmtree(path)

    assert len(os.listdir(processing_path)) == 0

    # Nothing should exist yet
    assert not os.path.exists(event_path)

    # Process some data.
    cli.run("lasif process_data GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")

    assert len(os.listdir(processing_path)) >= 1
    assert os.path.exists(event_path)
    # The test below should be changed as soon as we have some data
    assert len(os.listdir(event_path)) == 1


def test_processing_event_limiting_works(cli):
    """
    Asserts that the event parsing is correct.
    """
    ac = "lasif.components.actions.ActionsComponent."
    # cli.run("lasif create_new_iteration 1 8.0 100.0 SES3D_4_1")

    # No event should result in None.
    with mock.patch(ac + "process_data") as patch:
        cli.run("lasif process_data")
    assert patch.call_count == 1
    patch.assert_called_once_with(["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
                                   "GCMT_event_TURKEY_Mag_5.9"
                                   "_2011-5-19-20-15"])

    # One specified event should result in one event.
    with mock.patch(ac + "process_data") as patch:
        cli.run("lasif process_data "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11")
    assert patch.call_count == 1
    patch.assert_called_once_with(
        ["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"])

    # Multiple result in multiple.
    with mock.patch(ac + "process_data") as patch:
        cli.run("lasif process_data "
                "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11 "
                "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15")
    assert patch.call_count == 1
    patch.assert_called_once_with(["GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
                                   "GCMT_event_TURKEY_Mag_5.9_"
                                   "2011-5-19-20-15"])

    out = cli.run("lasif process_data blub wub").stdout
    assert "Event 'blub' not found." in out


def test_validate_data(cli):
    """
    Tests the validate_data command with mock and some real tests.
    """
    vc = "lasif.components.validator.ValidatorComponent."
    with mock.patch(vc + "validate_data") as patch:
        cli.run("lasif validate_data")
        patch.assert_called_once_with(data_and_station_file_availability=False,
                                      raypaths=False)

    with mock.patch(vc + "validate_data") as patch:
        cli.run("lasif validate_data --full")
        patch.assert_called_once_with(data_and_station_file_availability=True,
                                      raypaths=True)

    with mock.patch("lasif.domain.ExodusDomain.point_in_domain") as patch:
        cli.run("lasif validate_data")
        assert patch.call_count == 2

    # Have the raypath check fail.
    # Add this one when we have some data. Now it doesn't perform the ray check
    with mock.patch('lasif.components.validator.ValidatorComponent'
                    '.is_event_station_raypath_within_boundaries') as p:
        p.return_value = False
        out = cli.run("lasif validate_data --full")
        assert "Validating 2 event files" in out.stdout


def test_open_tutorial(cli):
    """
    Simple mock test.
    """
    with mock.patch("webbrowser.open") as patch:
        cli.run("lasif tutorial")
        patch.assert_called_once_with("http://dirkphilip.github.io/LASIF_2.0/")


def test_version_str(cli):
    """
    Tests if the version is printed correctly.
    """
    out = cli.run("lasif --version")
    assert out.stderr == ""
    assert out.stdout.strip() == "LASIF version %s" % lasif.__version__
