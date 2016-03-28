#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the window selection.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import obspy
import os

from lasif.window_selection import select_windows

# Data path.
DATA = os.path.join(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe()))),
    "data", "window_selection_test_files")


def test_select_windows():
    """
    Simple test against existing windows that are considered "good".
    """
    data_trace = obspy.read(os.path.join(DATA, "LA.AA10..BHZ.mseed"))[0]
    synthetic_trace = obspy.read(os.path.join(DATA, "LA.AA10_.___.z.bz2"))[0]

    event_latitude = 44.87
    event_longitude = 8.48
    event_depth_in_km = 15.0
    station_latitude = 41.3317000452
    station_longitude = 2.00073761549
    minimum_period = 40.0
    maximum_period = 100
    min_cc = 0.10
    max_noise = 0.10
    max_noise_window = 0.4
    min_velocity = 2.4
    threshold_shift = 0.30
    threshold_correlation = 0.75
    min_length_period = 1.5
    min_peaks_troughs = 2
    max_energy_ratio = 2.0

    windows = select_windows(
        data_trace=data_trace,
        synthetic_trace=synthetic_trace,
        event_latitude=event_latitude,
        event_longitude=event_longitude,
        event_depth_in_km=event_depth_in_km,
        station_latitude=station_latitude,
        station_longitude=station_longitude,
        minimum_period=minimum_period,
        maximum_period=maximum_period,
        min_cc=min_cc,
        max_noise=max_noise,
        max_noise_window=max_noise_window,
        min_velocity=min_velocity,
        threshold_shift=threshold_shift,
        threshold_correlation=threshold_correlation,
        min_length_period=min_length_period,
        min_peaks_troughs=min_peaks_troughs,
        max_energy_ratio=max_energy_ratio)

    expected_windows = [(obspy.UTCDateTime(2000, 8, 21, 17, 15, 38, 300000),
                         obspy.UTCDateTime(2000, 8, 21, 17, 19, 24, 800000))]

    assert windows == expected_windows
