#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project specific function picking windows.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from lasif.window_selection import select_windows


def window_picking_function(data_trace, synthetic_trace, event_latitude,
                            event_longitude, event_depth_in_km,
                            station_latitude, station_longitude,
                            minimum_period, maximum_period,
                            iteration, **kwargs):  # NOQA
    """
    Function that will be called every time a window is picked. This is part
    of the project so it can change depending on the project.

    Please keep in mind that you will have to manually update this file to a
    new version if LASIF is ever updated.

    You can do whatever you want in this function as long as the function
    signature is honored and the correct data types are returned. You could
    for example only tweak the window picking parameters but you could also
    implement your own window picking algorithm or call some other tool that
    picks windows.

    This function has to return a list of tuples of start and end times,
    each tuple denoting a selected window.

    :param data_trace: Trace containing the fully preprocessed data.
    :type data_trace: :class:`~obspy.core.trace.Trace`
    :param synthetic_trace: Trace containing the fully preprocessed synthetics.
    :type synthetic_trace: :class:`~obspy.core.trace.Trace`
    :param event_latitude: The event latitude.
    :type event_latitude: float
    :param event_longitude: The event longitude.
    :type event_longitude: float
    :param event_depth_in_km: The event depth in km.
    :type event_depth_in_km: float
    :param station_latitude: The station latitude.
    :type station_latitude: float
    :param station_longitude: The station longitude.
    :type station_longitude: float
    :param minimum_period: The minimum period of data and synthetics.
    :type minimum_period: float
    :param maximum_period: The maximum period of data and synthetics.
    :type maximum_period: float
    :param iteration: The iteration object. Use this to change behaviour based
        on the iteration.
    :type iteration: :class:`lasif.iteration-xml.Iteration`

    Please note that you also got the iteration object here, so if you
    want some parameters to change depending on the iteration, just use
    if/else on the iteration objects.

    >>> iteration.name  # doctest: +SKIP
    '11'
    >>> iteration.get_process_params()  # doctest: +SKIP
    {'dt': 0.75,
     'highpass': 0.01,
     'lowpass': 0.02,
     'npts': 500}

    Use ``$ lasif shell`` to play around and figure out what the iteration
    objects can do.
    """
    # Minimum normalised correlation coefficient of the complete traces.
    MIN_CC = 0.10

    # Maximum relative noise level for the whole trace. Measured from
    # maximum amplitudes before and after the first arrival.
    MAX_NOISE = 0.10

    # Maximum relative noise level for individual windows.
    MAX_NOISE_WINDOW = 0.4

    # All arrivals later than those corresponding to the threshold velocity
    # [km/s] will be excluded.
    MIN_VELOCITY = 2.4

    # Maximum allowable time shift within a window, as a fraction of the
    # minimum period.
    THRESHOLD_SHIFT = 0.30

    # Minimum normalised correlation coefficient within a window.
    THRESHOLD_CORRELATION = 0.75

    # Minimum length of the time windows relative to the minimum period.
    MIN_LENGTH_PERIOD = 1.5

    # Minimum number of extrema in an individual time window (excluding the
    # edges).
    MIN_PEAKS_TROUGHS = 2

    # Maximum energy ratio between data and synthetics within a time window.
    # Don't make this too small!
    MAX_ENERGY_RATIO = 10.0

    # The minimum similarity of the envelopes of both data and synthetics. This
    # essentially assures that the amplitudes of data and synthetics can not
    # diverge too much within a window. It is a bit like the inverse of the
    # ratio of both envelopes so a value of 0.2 makes sure neither amplitude
    # can be more then 5 times larger than the other.
    MIN_ENVELOPE_SIMILARITY = 0.2

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
        # User adjustable parameters.
        min_cc=MIN_CC,
        max_noise=MAX_NOISE,
        max_noise_window=MAX_NOISE_WINDOW,
        min_velocity=MIN_VELOCITY,
        threshold_shift=THRESHOLD_SHIFT,
        threshold_correlation=THRESHOLD_CORRELATION,
        min_length_period=MIN_LENGTH_PERIOD,
        min_peaks_troughs=MIN_PEAKS_TROUGHS,
        max_energy_ratio=MAX_ENERGY_RATIO,
        min_envelope_similarity=MIN_ENVELOPE_SIMILARITY,
        **kwargs)

    return windows
