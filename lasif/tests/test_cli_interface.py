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
from unittest import mock

import lasif
from lasif.scripts import lasif_cli
# from lasif.tests.testing_helpers import cli
from lasif.tests.testing_helpers import communicator, cli  # NOQA
# from lasif.tests.testing_helpers import reset_matplotlib
#
# Get a list of all available commands.
CMD_LIST = [key.replace("lasif_", "")
            for (key, value) in lasif_cli.__dict__.items()
            if (key.startswith("lasif_") and callable(value))]
#
#
# def setup_function(function):
#     """
#     Make sure matplotlib behaves the same on every machine.
#     """
#     reset_matplotlib()


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
    patch.assert_called_once_with(plot_simulation_domain=True)
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


# def test_lasif_info(cli):
#     """
#     Tests the 'lasif info' command.
#     """
#     out = cli.run("lasif info").stdout
#     assert "\"ExampleProject\"" in out
#     assert "Toy Project used in the Test Suite" in out
#     assert "2 events" in out
#     assert "4 station files" in out
#     assert "6 raw waveform files" in out
#     assert "0 processed waveform files" in out
#     assert "6 synthetic waveform files" in out
#
#
def test_various_list_functions(cli):
    """
    Tests all the "lasif list_" functions.
    """
    events = cli.run("lasif list_events").stdout
    assert "2 events" in events
    assert "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10" in events
    assert "GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_5.8_2014-10-15-13" in events

    # Also has a --list option.
    events = cli.run("lasif list_events --list").stdout
    assert events == (
        "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10\n"
        "GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_5.8_2014-10-15-13\n")
    #
    # # Bot arguments cannot be passed at the same time.
    # out = cli.run("lasif list_events --list --details").stdout
    # assert "--list and --details cannot both be specified." in out
    #
    iterations = cli.run("lasif list_iterations").stdout
    assert "no iterations in this project" in iterations
    # with open(os.path.join(cli.comm.project.paths["iterations"],
    #                        "ITERATION_1.xml"), "wt") as fh:
    #     fh.write("<>")
    # iterations = cli.run("lasif list_iterations").stdout
    # assert "1 iteration" in iterations
    # with open(os.path.join(cli.comm.project.paths["iterations"],
    #                        "ITERATION_2.xml"), "wt") as fh:
    #     fh.write("<>")
    # iterations = cli.run("lasif list_iterations").stdout
    # assert "2 iterations" in iterations
    #
    # models = cli.run("lasif list_models").stdout
    # assert "0 models" in models
    # os.makedirs(os.path.join(cli.comm.project.paths["models"], "BLUB"))
    # models = cli.run("lasif list_models").stdout
    # assert "1 model" in models


def test_iteration_creation(cli):
    """
    Tests the generation of an iteration and removal of one
    """
    cli.run("lasif set_up_iteration 1")
    assert cli.comm.iterations.has_iteration("1")
    cli.run("lasif set_up_iteration 1 --remove_dirs")
    assert not cli.comm.iterations.has_iteration("1")
    assert not cli.comm.iterations.has_iteration("2")
    # with mock.patch("lasif.visualization.plot_tf") as patch:
    #     cli.run("lasif plot_stf 1")
    # assert patch.call_count == 1
    # data, delta = patch.call_args[0]
    # np.testing.assert_array_equal(
    #     data,
    #     cli.comm.iterations.get("1").get_source_time_function()["data"])
    # assert delta == 0.75


def test_lasif_event_info(cli):
    """
    Tests the event info function.
    """
    event_1 = cli.run("lasif event_info "
                      "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10").stdout
    event_2 = cli.run("lasif event_info "
                      "GCMT_event_IRAN-IRAQ_BORDER_"
                      "REGION_Mag_5.8_2014-10-15-13").stdout

    assert "5.5 Mw" in event_1
    assert "ICELAND" in event_1
    assert "Latitude:" in event_1
    assert "Longitude:" in event_1
    assert "Depth:" in event_1
    assert "available at" in event_1

    assert "5.8 Mw" in event_2
    assert "IRAN-IRAQ BORDER REGION" in event_2
    assert "Latitude:" in event_2
    assert "Longitude:" in event_2
    assert "Depth:" in event_2
    assert "available at" in event_2


# def test_plot_stf(cli):
#     """
#     Tests the source time function plots.
#     """
#     with mock.patch("lasif.visualization.plot_tf") as patch:
#         cli.run("lasif plot_stf")
#     assert patch.call_count == 1
#     data = patch.call_args[0]
#     delta = patch.call_args[1]
#     stf_fct = cli.comm.project.get_project_function("source_time_function")
#     stf_delta = cli.comm.project.solver_settings["time_increment"]
#     stf_npts = cli.comm.project.solver_settings["number_of_time_steps"]
#     stf_freqmin = 1.0 / cli.comm.project.processing_params["lowpass_period"]
#     stf_freqmax = 1.0 / cli.comm.project.processing_params["highpass_period"]
#
#     stf_data = stf_fct(npts=stf_npts, delta=stf_delta, freqmin=stf_freqmin,
#                        freqmax=stf_freqmax)
#     np.testing.assert_array_equal(data, stf_data)
#     assert stf_delta == delta


def test_generate_input_files(cli):
    """
    Mock test for generate_all_input_files.
    """
    ac = "lasif.components.actions.ActionsComponent."
    event = "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10"
    # Test if it gives an error if there is no iteration available
    with mock.patch(ac + 'generate_input_files') as patch:
        out = cli.run("lasif generate_input_files 1 " + event + ' forward')
    assert "Could not find iteration: 1" in out.stderr
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
        out = cli.run("lasif calculate_adjoint_sources 1 A "
                      "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10")
    assert "Window set A not known to LASIF" in out.stderr
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
                      "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10 adjoint")
    assert out.stderr == ""
    p.assert_called_once_with(
        "1", "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10", None)
    assert p.call_count == 1


def test_launch_misfit_gui(cli):
    with mock.patch("lasif.misfit_gui.misfit_gui.launch") as patch:
        cli.run("lasif launch_misfit_gui")

    assert patch.call_count == 1


def test_preprocessing(cli):
    """
    Tests the processing.
    """
    event = "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10"
    processing_path = cli.comm.project.paths["preproc_eq_data"]
    event_path = os.path.join(processing_path, event)

    assert len(os.listdir(processing_path)) == 1

    # Nothing should exist yet
    assert not os.path.exists(event_path)

    # Process some data.
    cli.run("lasif process_data")

    assert len(os.listdir(processing_path)) > 1
    assert os.path.exists(event_path)
    # The test below should be changed as soon as we have some data
    assert len(os.listdir(event_path)) == 0


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
    patch.assert_called_once_with(["GCMT_event_ICELAND_Mag_5.5_2014-10-7-10",
                                   "GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_"
                                   "5.8_2014-10-15-13"])

    # One specified event should result in one event.
    with mock.patch(ac + "process_data") as patch:
        cli.run("lasif process_data "
                "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10")
    assert patch.call_count == 1
    patch.assert_called_once_with(
        ["GCMT_event_ICELAND_Mag_5.5_2014-10-7-10"])

    # Multiple result in multiple.
    with mock.patch(ac + "process_data") as patch:
        cli.run("lasif process_data "
                "GCMT_event_ICELAND_Mag_5.5_2014-10-7-10 "
                "GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag_5.8_2014-10-15-13")
    assert patch.call_count == 1
    patch.assert_called_once_with(["GCMT_event_ICELAND_Mag_5.5_2014-10-7-10",
                                   "GCMT_event_IRAN-IRAQ_BORDER_REGION_Mag"
                                   "_5.8_2014-10-15-13"])

    out = cli.run("lasif process_data blub wub").stdout
    assert "Event 'blub' not found." in out
#
#
# def test_remove_empty_coordinate_entries(cli):
#     """
#     Simple mock test.
#     """
#     with mock.patch("lasif.components.inventory_db.InventoryDBComponent"
#                     ".remove_coordinate_less_stations")\
#             as patch:
#         out = cli.run("lasif remove_empty_coordinate_entries")
#     assert out.stderr == ""
#     patch.assert_called_once_with()
#     assert patch.call_count == 1


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

    # Have the raypath check fail.
    # Add this one when we have some data. Now it doesn't perform the ray check
    # with mock.patch('lasif.components.validator.ValidatorComponent'
    #                 '.is_event_station_raypath_within_boundaries') as p:
    #     p.return_value = False
    #     out = cli.run("lasif validate_data --full")
    #    assert "Some files failed the raypath in domain checks." in out.stdout
    #     # Created script that deletes the extraneous files.
    #     filename = out.stdout.splitlines()[-1].strip().strip("'")
    #     assert os.path.exists(filename)
    #     assert filename.endswith("delete_raypath_violating_files.sh")
    #     lines = []
    #     with open(filename, "rt") as fh:
    #         for line in fh.readlines():
    #             if not line.startswith("rm"):
    #                 continue
    #             lines.append(line)
    #     # Make sure all 6 files are actually deleted.
    #     assert len(lines) == 6
#
#
# def test_open_tutorial(cli):
#     """
#     Simple mock test.
#     """
#     with mock.patch("webbrowser.open") as patch:
#         cli.run("lasif tutorial")
#         patch.assert_called_once_with("http://krischer.github.io/LASIF/")
#
#
# def test_iteration_status_command(cli):
#     """
#     The iteration status command returns the current state of any iteration.
#     It returns the number of already preprocessed data files,
#     how many synthetics are available, the windows and adjoint sources.
#     """
#     cli.run("lasif create_new_iteration 1 8.0 100.0 SES3D_4_1")
#     out = cli.run("lasif iteration_status 1").stdout.splitlines()
#     assert [_i.strip() for _i in out] == [
#         "Iteration 1 is defined for 1 events:",
#         "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
#         "0.00 % of the events stations have picked windows",
#         "Lacks processed data for 4 stations",
#         "Lacks synthetic data for 2 stations",
#     ]
#
#     cli.run("lasif preprocess_data 1")
#     out = cli.run("lasif iteration_status 1").stdout.splitlines()
#     assert [_i.strip() for _i in out] == [
#         "Iteration 1 is defined for 1 events:",
#         "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
#         "0.00 % of the events stations have picked windows",
#         "Lacks synthetic data for 2 stations",
#     ]
#
#     # Copy the data for the first event to the second.
#     shutil.rmtree(os.path.join(
#         cli.comm.project.paths["data"],
#         "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"))
#     shutil.copytree(
#         os.path.join(cli.comm.project.paths["data"],
#                      "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"),
#         os.path.join(cli.comm.project.paths["data"],
#                      "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15"))
#     # The iteration has to be recreated.
#     os.remove(os.path.join(cli.comm.project.paths["iterations"],
#                            "ITERATION_1.xml"))
#
#     cli.run("lasif create_new_iteration 1 8.0 100.0 SES3D_4_1")
#     out = cli.run("lasif iteration_status 1").stdout.splitlines()
#     assert [_i.strip() for _i in out] == [
#         "Iteration 1 is defined for 2 events:",
#         "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11",
#         "0.00 % of the events stations have picked windows",
#         "Lacks synthetic data for 2 stations",
#         "GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15",
#         "0.00 % of the events stations have picked windows",
#         "Lacks synthetic data for 4 stations",
#     ]
#
#
# def test_debug_information(cli):
#     """
#     Tests the debugging information.
#     """
#     # Files not found.
#     out = cli.run("lasif debug DUMMY_1 DUMMY_2").stdout
#     assert "Path 'DUMMY_1' does not exist." in out
#     assert "Path 'DUMMY_2' does not exist." in out
#
#     # Check a file to make sure the binding works. Other file types are
#     # tested elsewhere.
#     out = cli.run("lasif debug " + cli.comm.project.paths["config_file"])
#     assert "The main project configuration file" in out.stdout


def test_version_str(cli):
    """
    Tests if the version is printed correctly.
    """
    out = cli.run("lasif --version")
    assert out.stderr == ""
    assert out.stdout.strip() == "LASIF version %s" % lasif.__version__
#
#
# def test_lasif_serve(cli):
#     """
#     Tests that the correct serve functions are called.
#     """
#     # The lambda function in the timer is never executed, thus does not
#     # require to be mocked.
#     with mock.patch("lasif.webinterface.server.serve") as serve_patch:
#         with mock.patch("threading.Timer") as timer_patch:
#             cli.run("lasif serve")
#             assert serve_patch.call_count == 1
#             assert timer_patch.call_count == 1
#             assert len(serve_patch.call_args[0]) == 1
#             assert len(serve_patch.call_args[1]) == 3
#             assert serve_patch.call_args[1]["port"] == 8008
#             assert serve_patch.call_args[1]["debug"] is False
#             assert serve_patch.call_args[1]["open_to_outside"] is False
#             serve_patch.reset_mock()
#             timer_patch.reset_mock()
#
#             cli.run("lasif serve --port=9999")
#             assert serve_patch.call_count == 1
#             assert timer_patch.call_count == 1
#             assert len(serve_patch.call_args[0]) == 1
#             assert len(serve_patch.call_args[1]) == 3
#             assert serve_patch.call_args[1]["port"] == 9999
#             assert serve_patch.call_args[1]["debug"] is False
#             assert serve_patch.call_args[1]["open_to_outside"] is False
#             serve_patch.reset_mock()
#             timer_patch.reset_mock()
#
#             cli.run("lasif serve --port=9999 --debug")
#             assert serve_patch.call_count == 1
#             # Debug settings turn of opening the browser.
#             assert timer_patch.call_count == 0
#             assert len(serve_patch.call_args[0]) == 1
#             assert len(serve_patch.call_args[1]) == 3
#             assert serve_patch.call_args[1]["port"] == 9999
#             assert serve_patch.call_args[1]["debug"] is True
#             assert serve_patch.call_args[1]["open_to_outside"] is False
#             serve_patch.reset_mock()
#             timer_patch.reset_mock()
#
#             cli.run("lasif serve --port=9999 --nobrowser")
#             assert serve_patch.call_count == 1
#             assert timer_patch.call_count == 0
#             assert len(serve_patch.call_args[0]) == 1
#             assert len(serve_patch.call_args[1]) == 3
#             assert serve_patch.call_args[1]["port"] == 9999
#             assert serve_patch.call_args[1]["debug"] is False
#             assert serve_patch.call_args[1]["open_to_outside"] is False
#             serve_patch.reset_mock()
#             timer_patch.reset_mock()
#
#             cli.run("lasif serve --port=9999 --nobrowser --open_to_outside")
#             assert serve_patch.call_count == 1
#             assert timer_patch.call_count == 0
#             assert len(serve_patch.call_args[0]) == 1
#             assert len(serve_patch.call_args[1]) == 3
#             assert serve_patch.call_args[1]["port"] == 9999
#             assert serve_patch.call_args[1]["debug"] is False
#             assert serve_patch.call_args[1]["open_to_outside"] is True
#             serve_patch.reset_mock()
#             timer_patch.reset_mock()
