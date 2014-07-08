#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the parallel helper methods.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from obspy import UTCDateTime
import os
import pytest

from lasif.window_manager import Window, WindowCollection


def test_window_class():
    """
    Tests the window class. Not much to test but better then nothing.
    """
    win = Window(
        starttime=UTCDateTime(2012, 1, 1),
        endtime=UTCDateTime(2012, 1, 1, 0, 1),
        weight=1.0,
        taper="cosine",
        taper_percentage=0.05,
        misfit="RandomMisfit",
        misfit_value=1E-5)

    assert win.duration == 60.0

    assert str(win) == (
        "60.00 seconds window from 2012-01-01T00:00:00.000000Z\n"
        "\tWeight: 1.00, 5.00% cosine taper, RandomMisfit (1e-05)")


def test_window_collection_init_failures(tmpdir):
    """
    Tests initializing the window collection class.
    """
    tmpdir = str(tmpdir)
    filename = os.path.join(tmpdir, "random.xml")

    # Initializing the collection with a non-existing filename and no
    # event_name, channel_id, or synthetics_tag raises an error.
    with pytest.raises(ValueError):
        WindowCollection(filename=filename)
    with pytest.raises(ValueError):
        WindowCollection(filename=filename, event_name="A")
    with pytest.raises(ValueError):
        WindowCollection(filename=filename, event_name="A",
                         channel_id="A.B.C.D")
    with pytest.raises(ValueError):
        WindowCollection(filename=filename, synthetics_tag="ABCD")

    # Initializing with an existing file and windows also raises.
    with open(filename, "wb") as fh:
        fh.write("")
    with pytest.raises(ValueError) as error:
        WindowCollection(filename=filename, windows=[1, 2, 3])
    assert "existing file and new windows" in str(error.value).lower()

def test_window_collection_i_o(tmpdir):
    """
    Reading and writing windows.
    """
    tmpdir = str(tmpdir)
    filename = os.path.join(tmpdir, "random.xml")

    wc = WindowCollection(filename, event_name="SomeEvent",
                          channel_id="AA.BB.CC.DD",
                          synthetics_tag="Iteration_A")
    wc.add_window(starttime=UTCDateTime(2012, 1, 1),
                  endtime=UTCDateTime(2012, 1, 1, 0, 1),
                  weight=0.5, taper="cosine", taper_percentage=0.08,
                  misfit="some misfit", misfit_value=1E-5)
    wc.write()

    # Hardcode an example to be sure the format is correct.
    reference = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<MisfitWindow>\n"
        "  <Event>SomeEvent</Event>\n"
        "  <ChannelID>AA.BB.CC.DD</ChannelID>\n"
        "  <SyntheticsTag>Iteration_A</SyntheticsTag>\n"
        "  <Window>\n"
        "    <Starttime>2012-01-01T00:00:00.000000Z</Starttime>\n"
        "    <Endtime>2012-01-01T00:01:00.000000Z</Endtime>\n"
        "    <Weight>0.5</Weight>\n"
        "    <Taper>cosine</Taper>\n"
        "    <TaperPercentage>0.08</TaperPercentage>\n"
        "    <Misfit>some misfit</Misfit>\n"
        "    <MisfitValue>1e-05</MisfitValue>\n"
        "  </Window>\n"
        "</MisfitWindow>")
    with open(filename, "rb") as fh:
        actual = fh.read().strip()
    assert reference == actual

    # Read it once again and make sure it is identical.
    wc2 = WindowCollection(filename)
    assert wc == wc2

    # Add a new window and test the same again.
    wc.add_window(starttime=UTCDateTime(2013, 1, 1),
                  endtime=UTCDateTime(2013, 1, 1, 0, 1),
                  weight=0.8, taper="hanning", taper_percentage=0.1,
                  misfit="other misfit", misfit_value=0.0)
    wc.write()
    reference = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<MisfitWindow>\n"
        "  <Event>SomeEvent</Event>\n"
        "  <ChannelID>AA.BB.CC.DD</ChannelID>\n"
        "  <SyntheticsTag>Iteration_A</SyntheticsTag>\n"
        "  <Window>\n"
        "    <Starttime>2012-01-01T00:00:00.000000Z</Starttime>\n"
        "    <Endtime>2012-01-01T00:01:00.000000Z</Endtime>\n"
        "    <Weight>0.5</Weight>\n"
        "    <Taper>cosine</Taper>\n"
        "    <TaperPercentage>0.08</TaperPercentage>\n"
        "    <Misfit>some misfit</Misfit>\n"
        "    <MisfitValue>1e-05</MisfitValue>\n"
        "  </Window>\n"
        "  <Window>\n"
        "    <Starttime>2013-01-01T00:00:00.000000Z</Starttime>\n"
        "    <Endtime>2013-01-01T00:01:00.000000Z</Endtime>\n"
        "    <Weight>0.8</Weight>\n"
        "    <Taper>hanning</Taper>\n"
        "    <TaperPercentage>0.1</TaperPercentage>\n"
        "    <Misfit>other misfit</Misfit>\n"
        "    <MisfitValue>0.0</MisfitValue>\n"
        "  </Window>\n"
        "</MisfitWindow>")
    with open(filename, "rb") as fh:
        actual = fh.read().strip()
    assert reference == actual

    wc3 = WindowCollection(filename)
    assert wc == wc3
