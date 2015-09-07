#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Window selection algorithm.

This module aims to provide a window selection algorithm suitable for
calculating phase misfits between two seismic waveforms.

The main function is the select_windows() function. The selection process is a
multi-stage process. Initially all time steps are considered to be valid in
the sense as being suitable for window selection. Then a number of selectors
is applied, progressively excluding more and more time steps.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import itertools
import math

import numpy as np
from obspy.core.util import geodetics
import obspy.signal.filter
from scipy.signal import argrelextrema


def flatnotmasked_contiguous(time_windows):
    """
    Helper function enabling to loop over empty time windows.
    """
    fc = np.ma.flatnotmasked_contiguous(time_windows)
    # If nothing could be found, set the mask to true (which should already
    # be the case).
    if fc is None:
        return []
    else:
        return fc


def find_local_extrema(data):
    """
    Function finding local extrema. It can also deal with flat extrema,
    e.g. a flat top or bottom. In that case the first index of all flat
    values will be returned.

    Returns a tuple of maxima and minima indices.
    """
    length = len(data) - 1
    diff = np.diff(data)
    flats = np.argwhere(diff == 0)

    # Discard neighbouring flat points.
    new_flats = list(flats[0:1])
    for i, j in zip(flats[:-1], flats[1:]):
        if j - i == 1:
            continue
        new_flats.append(j)
    flats = new_flats

    maxima = []
    minima = []

    # Go over each flats position and check if its a maxima/minima.
    for idx in flats:
        l_type = "left"
        r_type = "right"
        for i in itertools.count():
            this_idx = idx - i - 1
            if diff[this_idx] < 0:
                l_type = "minima"
                break
            elif diff[this_idx] > 0:
                l_type = "maxima"
                break
        for i in itertools.count():
            this_idx = idx + i + 1
            if this_idx >= len(diff):
                break
            if diff[this_idx] < 0:
                r_type = "maxima"
                break
            elif diff[this_idx] > 0:
                r_type = "minima"
                break
        if r_type != l_type:
            continue
        if r_type == "maxima":
            maxima.append(int(idx))
        else:
            minima.append(int(idx))

    maxs = set(list(argrelextrema(data, np.greater)[0]))
    mins = set(list(argrelextrema(data, np.less)[0]))

    peaks, troughs = (
        sorted(list(maxs.union(set(maxima)))),
        sorted(list(mins.union(set(minima)))))

    # Special case handling for missing one or the other.
    if not peaks and not troughs:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    elif not peaks:
        if 0 not in troughs:
            peaks.insert(0, 0)
        if length not in troughs:
            peaks.append(length)
        return (np.array(peaks, dtype=np.int32),
                np.array(troughs, dtype=np.int32))
    elif not troughs:
        if 0 not in peaks:
            troughs.insert(0, 0)
        if length not in peaks:
            troughs.append(length)
        return (np.array(peaks, dtype=np.int32),
                np.array(troughs, dtype=np.int32))

    # Mark the first and last values as well to facilitate the peak and
    # trough marching algorithm
    if 0 not in peaks and 0 not in troughs:
        if peaks[0] < troughs[0]:
            troughs.insert(0, 0)
        else:
            peaks.insert(0, 0)
    if length not in peaks and length not in troughs:
        if peaks[-1] < troughs[-1]:
            peaks.append(length)
        else:
            troughs.append(length)

    return (np.array(peaks, dtype=np.int32),
            np.array(troughs, dtype=np.int32))


def find_closest(ref_array, target):
    """
    For every value in target, find the index of ref_array to which
    the value is closest.

    from http://stackoverflow.com/a/8929827/1657047

    :param ref_array: The reference array. Must be sorted!
    :type ref_array: :class:`numpy.ndarray`
    :param target: The target array.
    :type target: :class:`numpy.ndarray`

    >>> ref_array = np.arange(0, 20.)
    >>> target = np.array([-2, 100., 2., 2.4, 2.5, 2.6])
    >>> find_closest(ref_array, target)
    array([ 0, 19,  2,  2,  3,  3])
    """
    # A must be sorted
    idx = ref_array.searchsorted(target)
    idx = np.clip(idx, 1, len(ref_array) - 1)
    left = ref_array[idx - 1]
    right = ref_array[idx]
    idx -= target - left < right - target
    return idx


def _plot_mask(new_mask, old_mask, name=None):
    """
    Helper function plotting the remaining time segments after an elimination
    stage.

    Useful to figure out which stage is responsible for a certain window
    being picked/rejected.

    :param new_mask: The mask after the elimination stage.
    :param old_mask: The mask before the elimination stage.
    :param name: The name of the elimination stage.
    :return:
    """
    # Lazy imports as not needed by default.
    import matplotlib.pylab as plt  # NOQA
    import matplotlib.patheffects as PathEffects  # NOQA

    old_mask = old_mask.copy()
    new_mask = new_mask.copy()

    new_mask.mask = np.bitwise_xor(old_mask.mask, new_mask.mask)
    old_mask.mask = np.invert(old_mask.mask)
    for i in flatnotmasked_contiguous(old_mask):
        plt.fill_between((i.start, i.stop), (-1.0, -1.0), (2.0, 2.0),
                         color="gray", alpha=0.3, lw=0)

    new_mask.mask = np.invert(new_mask.mask)
    for i in flatnotmasked_contiguous(new_mask):
        plt.fill_between((i.start, i.stop), (-1.0, -1.0), (2.0, 2.0),
                         color="#fb9a99", lw=0)

    if name:
        plt.text(len(new_mask) - 1 - 20, 0.5, name, verticalalignment="center",
                 horizontalalignment="right",
                 path_effects=[
                     PathEffects.withStroke(linewidth=3, foreground="white")],
                 fontweight=500)
    plt.xlim(0, len(new_mask) - 1)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.gca().xaxis.set_ticklabels([])


def _window_generator(data_length, window_width):
    """
    Simple generator yielding start and stop indices for sliding windows.

    :param data_length: The complete length of the data series over which to
        slide the window.
    :param window_width: The desired window width.
    """
    window_start = 0
    while True:
        window_end = window_start + window_width
        if window_end > data_length:
            break
        yield (window_start, window_end, window_start + window_width // 2)
        window_start += 1


def _log_window_selection(tr_id, msg):
    """
    Helper function for consistent output during the window selection.

    :param tr_id: The id of the current trace.
    :param msg: The message to be printed.
    """
    print "[Window selection for %s] %s" % (tr_id, msg)

# Dictionary to cache the TauPyModel so there is no need to reinitialize it
# each time which is a fairly expensive operation.
TAUPY_MODEL_CACHE = {}


def select_windows(data_trace, synthetic_trace, event_latitude,
                   event_longitude, event_depth_in_km,
                   station_latitude, station_longitude, minimum_period,
                   maximum_period,
                   min_cc=0.10, max_noise=0.10, max_noise_window=0.4,
                   min_velocity=2.4, threshold_shift=0.30,
                   threshold_correlation=0.75, min_length_period=1.5,
                   min_peaks_troughs=2, max_energy_ratio=10.0,
                   min_envelope_similarity=0.2,
                   verbose=False, plot=False):
    """
    Window selection algorithm for picking windows suitable for misfit
    calculation based on phase differences.

    Returns a list of windows which might be empty due to various reasons.

    This function is really long and a lot of things. For a more detailed
    description, please see the LASIF paper.

    :param data_trace: The data trace.
    :type data_trace: :class:`~obspy.core.trace.Trace`
    :param synthetic_trace: The synthetic trace.
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
    :param minimum_period: The minimum period of the data in seconds.
    :type minimum_period: float
    :param maximum_period: The maximum period of the data in seconds.
    :type maximum_period: float
    :param min_cc: Minimum normalised correlation coefficient of the
        complete traces.
    :type min_cc: float
    :param max_noise: Maximum relative noise level for the whole trace.
        Measured from maximum amplitudes before and after the first arrival.
    :type max_noise: float
    :param max_noise_window: Maximum relative noise level for individual
        windows.
    :type max_noise_window: float
    :param min_velocity: All arrivals later than those corresponding to the
        threshold velocity [km/s] will be excluded.
    :type min_velocity: float
    :param threshold_shift: Maximum allowable time shift within a window,
        as a fraction of the minimum period.
    :type threshold_shift: float
    :param threshold_correlation: Minimum normalised correlation coeeficient
        within a window.
    :type threshold_correlation: float
    :param min_length_period: Minimum length of the time windows relative to
        the minimum period.
    :type min_length_period: float
    :param min_peaks_troughs: Minimum number of extrema in an individual
        time window (excluding the edges).
    :type min_peaks_troughs: float
    :param max_energy_ratio: Maximum energy ratio between data and
        synthetics within a time window. Don't make this too small!
    :type max_energy_ratio: float
    :param min_envelope_similarity: The minimum similarity of the envelopes of
        both data and synthetics. This essentially assures that the
        amplitudes of data and synthetics can not diverge too much within a
        window. It is a bit like the inverse of the ratio of both envelopes
        so a value of 0.2 makes sure neither amplitude can be more then 5
        times larger than the other.
    :type min_envelope_similarity: float
    :param verbose: No output by default.
    :type verbose: bool
    :param plot: Create a plot of the algortihm while it does its work.
    :type plot: bool
    """
    # Shortcuts to frequently accessed variables.
    data_starttime = data_trace.stats.starttime
    data_delta = data_trace.stats.delta
    dt = data_trace.stats.delta
    npts = data_trace.stats.npts
    synth = synthetic_trace.data
    data = data_trace.data
    times = data_trace.times()

    # Fill cache if necessary.
    if not TAUPY_MODEL_CACHE:
        from obspy.taup import TauPyModel  # NOQA
        TAUPY_MODEL_CACHE["model"] = TauPyModel("AK135")
    model = TAUPY_MODEL_CACHE["model"]

    # -------------------------------------------------------------------------
    # Geographical calculations and the time of the first arrival.
    # -------------------------------------------------------------------------
    dist_in_deg = geodetics.locations2degrees(station_latitude,
                                              station_longitude,
                                              event_latitude, event_longitude)
    dist_in_km = geodetics.calcVincentyInverse(
        station_latitude, station_longitude, event_latitude,
        event_longitude)[0] / 1000.0

    # Get only a couple of P phases which should be the first arrival
    # for every epicentral distance. Its quite a bit faster than calculating
    # the arrival times for every phase.
    # Assumes the first sample is the centroid time of the event.
    tts = model.get_travel_times(source_depth_in_km=event_depth_in_km,
                                 distance_in_degree=dist_in_deg,
                                 phase_list=["ttp"])
    # Sort just as a safety measure.
    tts = sorted(tts, key=lambda x: x.time)
    first_tt_arrival = tts[0].time

    # -------------------------------------------------------------------------
    # Window settings
    # -------------------------------------------------------------------------
    # Number of samples in the sliding window. Currently, the length of the
    # window is set to a multiple of the dominant period of the synthetics.
    # Make sure it is an uneven number; just to have a trivial midpoint
    # definition and one sample does not matter much in any case.
    window_length = int(round(float(2 * minimum_period) / dt))
    if not window_length % 2:
        window_length += 1

    # Use a Hanning window. No particular reason for it but its a well-behaved
    # window and has nice spectral properties.
    taper = np.hanning(window_length)

    # =========================================================================
    # check if whole seismograms are sufficiently correlated and estimate
    # noise level
    # =========================================================================

    # Overall Correlation coefficient.
    norm = np.sqrt(np.sum(data ** 2)) * np.sqrt(np.sum(synth ** 2))
    cc = np.sum(data * synth) / norm
    if verbose:
        _log_window_selection(data_trace.id,
                              "Correlation Coefficient: %.4f" % cc)

    # Estimate noise level from waveforms prior to the first arrival.
    idx_end = int(np.ceil((first_tt_arrival - 0.5 * minimum_period) / dt))
    idx_end = max(10, idx_end)
    idx_start = int(np.ceil((first_tt_arrival - 2.5 * minimum_period) / dt))
    idx_start = max(10, idx_start)

    if idx_start >= idx_end:
        idx_start = max(0, idx_end - 10)

    abs_data = np.abs(data)
    noise_absolute = abs_data[idx_start:idx_end].max()
    noise_relative = noise_absolute / abs_data.max()

    if verbose:
        _log_window_selection(data_trace.id,
                              "Absolute Noise Level: %e" % noise_absolute)
        _log_window_selection(data_trace.id,
                              "Relative Noise Level: %e" % noise_relative)

    # Basic global rejection criteria.
    accept_traces = True
    if (cc < min_cc) and (noise_relative > max_noise / 3.0):
        msg = "Correlation %.4f is below threshold of %.4f" % (cc, min_cc)
        if verbose:
            _log_window_selection(data_trace.id, msg)
        accept_traces = msg

    if noise_relative > max_noise:
        msg = "Noise level %.3f is above threshold of %.3f" % (
            noise_relative, max_noise)
        if verbose:
            _log_window_selection(
                data_trace.id, msg)
        accept_traces = msg

    # Calculate the envelope of both data and synthetics. This is to make sure
    # that the amplitude of both is not too different over time and is
    # used as another selector. Only calculated if the trace is generally
    # accepted as it is fairly slow.
    if accept_traces is True:
        data_env = obspy.signal.filter.envelope(data)
        synth_env = obspy.signal.filter.envelope(synth)

    # -------------------------------------------------------------------------
    # Initial Plot setup.
    # -------------------------------------------------------------------------
    # All the plot calls are interleaved. I realize this is really ugly but
    # the alternative would be to either have two functions (one with plots,
    # one without) or split the plotting function in various subfunctions,
    # neither of which are acceptable in my opinion. The impact on
    # performance is minimal if plotting is turned off: all imports are lazy
    # and a couple of conditionals are cheap.
    if plot:
        import matplotlib.pylab as plt  # NOQA
        import matplotlib.patheffects as PathEffects  # NOQA

        if accept_traces is True:
            plt.figure(figsize=(18, 12))
            plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.95,
                                wspace=None, hspace=0.0)
            grid = (31, 1)

            # Axes showing the data.
            data_plot = plt.subplot2grid(grid, (0, 0), rowspan=8)
        else:
            # Only show one axes it the traces are not accepted.
            plt.figure(figsize=(18, 3))

        # Plot envelopes if needed.
        if accept_traces is True:
            plt.plot(times, data_env, color="black", alpha=0.5, lw=0.4,
                     label="data envelope")
            plt.plot(synthetic_trace.times(), synth_env, color="#e41a1c",
                     alpha=0.4, lw=0.5, label="synthetics envelope")

        plt.plot(times, data, color="black", label="data", lw=1.5)
        plt.plot(synthetic_trace.times(), synth, color="#e41a1c",
                 label="synthetics",  lw=1.5)

        # Symmetric around y axis.
        middle = data.mean()
        d_max, d_min = data.max(), data.min()
        r = max(d_max - middle, middle - d_min) * 1.1
        ylim = (middle - r, middle + r)
        xlim = (times[0], times[-1])
        plt.ylim(*ylim)
        plt.xlim(*xlim)

        offset = (xlim[1] - xlim[0]) * 0.005
        plt.vlines(first_tt_arrival, ylim[0], ylim[1], colors="#ff7f00", lw=2)
        plt.text(first_tt_arrival + offset,
                 ylim[1] - (ylim[1] - ylim[0]) * 0.02,
                 "first arrival", verticalalignment="top",
                 horizontalalignment="left", color="#ee6e00",
                 path_effects=[
                     PathEffects.withStroke(linewidth=3, foreground="white")])

        plt.vlines(first_tt_arrival - minimum_period / 2.0, ylim[0], ylim[1],
                   colors="#ff7f00", lw=2)
        plt.text(first_tt_arrival - minimum_period / 2.0 - offset,
                 ylim[0] + (ylim[1] - ylim[0]) * 0.02,
                 "first arrival - min period / 2", verticalalignment="bottom",
                 horizontalalignment="right", color="#ee6e00",
                 path_effects=[
                     PathEffects.withStroke(linewidth=3, foreground="white")])

        for velocity in [6, 5, 4, 3, min_velocity]:
            tt = dist_in_km / velocity
            plt.vlines(tt, ylim[0], ylim[1], colors="gray", lw=2)
            if velocity == min_velocity:
                hal = "right"
                o_s = -1.0 * offset
            else:
                hal = "left"
                o_s = offset
            plt.text(tt + o_s, ylim[0] + (ylim[1] - ylim[0]) * 0.02,
                     str(velocity) + " km/s", verticalalignment="bottom",
                     horizontalalignment=hal, color="0.15")
        plt.vlines(dist_in_km / min_velocity + minimum_period / 2.0,
                   ylim[0], ylim[1], colors="gray", lw=2)
        plt.text(dist_in_km / min_velocity + minimum_period / 2.0 - offset,
                 ylim[1] - (ylim[1] - ylim[0]) * 0.02,
                 "min surface velocity + min period / 2",
                 verticalalignment="top",
                 horizontalalignment="right", color="0.15", path_effects=[
                     PathEffects.withStroke(linewidth=3, foreground="white")])

        plt.hlines(noise_absolute, xlim[0], xlim[1], linestyle="--",
                   color="gray")
        plt.hlines(-noise_absolute, xlim[0], xlim[1], linestyle="--",
                   color="gray")
        plt.text(offset, noise_absolute + (ylim[1] - ylim[0]) * 0.01,
                 "noise level", verticalalignment="bottom",
                 horizontalalignment="left", color="0.15",
                 path_effects=[
                     PathEffects.withStroke(linewidth=3, foreground="white")])
        plt.legend(loc="lower right", fancybox=True, framealpha=0.5,
                   fontsize="small")
        plt.gca().xaxis.set_ticklabels([])

        # Plot the basic global information.
        ax = plt.gca()
        txt = (
            "Total CC Coeff: %.4f\nAbsolute Noise: %e\nRelative Noise: %.3f"
            % (cc, noise_absolute, noise_relative))
        ax.text(0.01, 0.95, txt, transform=ax.transAxes,
                fontdict=dict(fontsize="small", ha='left', va='top'),
                bbox=dict(boxstyle="round", fc="w", alpha=0.8))
        plt.suptitle("Channel %s" % data_trace.id, fontsize="larger")

    # Show plot and return if not accepted.
        if accept_traces is not True:
            txt = "Rejected: %s" % (accept_traces)
            ax.text(0.99, 0.95, txt, transform=ax.transAxes,
                    fontdict=dict(fontsize="small", ha='right', va='top'),
                    bbox=dict(boxstyle="round", fc="red", alpha=1.0))
            plt.show()
    if accept_traces is not True:
        return []

    # Initialise masked arrays. The mask will be set to True where no
    # windows are chosen.
    time_windows = np.ma.ones(npts)
    time_windows.mask = False
    if plot:
        old_time_windows = time_windows.copy()

    # Elimination Stage 1: Eliminate everything half a period before or
    # after the minimum and maximum travel times, respectively.
    # theoretical arrival as positive.
    min_idx = int((first_tt_arrival - (minimum_period / 2.0)) / dt)
    max_idx = int(math.ceil((
        dist_in_km / min_velocity + minimum_period / 2.0) / dt))
    time_windows.mask[:min_idx + 1] = True
    time_windows.mask[max_idx:] = True
    if plot:
        plt.subplot2grid(grid, (8, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="TRAVELTIME ELIMINATION")
        old_time_windows = time_windows.copy()

    # -------------------------------------------------------------------------
    # Compute sliding time shifts and correlation coefficients for time
    # frames that passed the traveltime elimination stage.
    # -------------------------------------------------------------------------
    # Allocate arrays to collect the time dependent values.
    sliding_time_shift = np.ma.zeros(npts, dtype="float32")
    sliding_time_shift.mask = True
    max_cc_coeff = np.ma.zeros(npts, dtype="float32")
    max_cc_coeff.mask = True

    for start_idx, end_idx, midpoint_idx in _window_generator(npts,
                                                              window_length):
        if not min_idx < midpoint_idx < max_idx:
            continue

        # Slice windows. Create a copy to be able to taper without affecting
        # the original time series.
        data_window = data[start_idx: end_idx].copy() * taper
        synthetic_window = \
            synth[start_idx: end_idx].copy() * taper

        # Elimination Stage 2: Skip windows that have essentially no energy
        # to avoid instabilities. No windows can be picked in these.
        if synthetic_window.ptp() < synth.ptp() * 0.001:
            time_windows.mask[midpoint_idx] = True
            continue

        # Calculate the time shift. Here this is defined as the shift of the
        # synthetics relative to the data. So a value of 2, for instance, means
        # that the synthetics are 2 timesteps later then the data.
        cc = np.correlate(data_window, synthetic_window, mode="full")

        time_shift = cc.argmax() - window_length + 1
        # Express the time shift in fraction of the minimum period.
        sliding_time_shift[midpoint_idx] = (time_shift * dt) / minimum_period

        # Normalized cross correlation.
        max_cc_value = cc.max() / np.sqrt((synthetic_window ** 2).sum() *
                                          (data_window ** 2).sum())
        max_cc_coeff[midpoint_idx] = max_cc_value

    if plot:
        plt.subplot2grid(grid, (9, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="NO ENERGY IN CC WINDOW")
        # Axes with the CC coeffs
        plt.subplot2grid(grid, (15, 0), rowspan=4)
        plt.hlines(0, xlim[0], xlim[1], color="lightgray")
        plt.hlines(-threshold_shift, xlim[0], xlim[1], color="gray",
                   linestyle="--")
        plt.hlines(threshold_shift, xlim[0], xlim[1], color="gray",
                   linestyle="--")
        plt.text(5, -threshold_shift - (2) * 0.03,
                 "threshold", verticalalignment="top",
                 horizontalalignment="left", color="0.15",
                 path_effects=[
                     PathEffects.withStroke(linewidth=3, foreground="white")])
        plt.plot(times, sliding_time_shift, color="#377eb8",
                 label="Time shift in fraction of minimum period", lw=1.5)
        ylim = plt.ylim()
        plt.yticks([-0.75, 0, 0.75])
        plt.xticks([300, 600, 900, 1200, 1500, 1800])
        plt.ylim(ylim[0], ylim[1] + ylim[1] - ylim[0])
        plt.ylim(-1.0, 1.0)
        plt.xlim(xlim)
        plt.gca().xaxis.set_ticklabels([])
        plt.legend(loc="lower right", fancybox=True, framealpha=0.5,
                   fontsize="small")

        plt.subplot2grid(grid, (10, 0), rowspan=4)
        plt.hlines(threshold_correlation, xlim[0], xlim[1], color="0.15",
                   linestyle="--")
        plt.hlines(1, xlim[0], xlim[1], color="lightgray")
        plt.hlines(0, xlim[0], xlim[1], color="lightgray")
        plt.text(5, threshold_correlation + (1.4) * 0.01,
                 "threshold", verticalalignment="bottom",
                 horizontalalignment="left", color="0.15",
                 path_effects=[
                     PathEffects.withStroke(linewidth=3, foreground="white")])
        plt.plot(times, max_cc_coeff, color="#4daf4a",
                 label="Maximum CC coefficient", lw=1.5)
        plt.ylim(-0.2, 1.2)
        plt.yticks([0, 0.5, 1])
        plt.xticks([300, 600, 900, 1200, 1500, 1800])
        plt.xlim(xlim)
        plt.gca().xaxis.set_ticklabels([])
        plt.legend(loc="lower right", fancybox=True, framealpha=0.5,
                   fontsize="small")

    # Elimination Stage 3: Mark all areas where the normalized cross
    # correlation coefficient is under threshold_correlation as negative
    if plot:
        old_time_windows = time_windows.copy()
    time_windows.mask[max_cc_coeff < threshold_correlation] = True
    if plot:
        plt.subplot2grid(grid, (14, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="CORRELATION COEFF THRESHOLD ELIMINATION")

    # Elimination Stage 4: Mark everything with an absolute travel time
    # shift of more than # threshold_shift times the dominant period as
    # negative
    if plot:
        old_time_windows = time_windows.copy()
    time_windows.mask[np.ma.abs(sliding_time_shift) > threshold_shift] = True
    if plot:
        plt.subplot2grid(grid, (19, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="TIME SHIFT THRESHOLD ELIMINATION")

    # Elimination Stage 5: Mark the area around every "travel time shift
    # jump" (based on the traveltime time difference) negative. The width of
    # the area is currently chosen to be a tenth of a dominant period to
    # each side.
    if plot:
        old_time_windows = time_windows.copy()
    sample_buffer = int(np.ceil(minimum_period / dt * 0.1))
    indices = np.ma.where(np.ma.abs(np.ma.diff(sliding_time_shift)) > 0.1)[0]
    for index in indices:
        time_windows.mask[index - sample_buffer: index + sample_buffer] = True
    if plot:
        plt.subplot2grid(grid, (20, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="TIME SHIFT JUMPS ELIMINATION")

    # Clip both to avoid large numbers by division.
    stacked = np.vstack([
        np.ma.clip(synth_env, synth_env.max() * min_envelope_similarity * 0.5,
                   synth_env.max()),
        np.ma.clip(data_env, data_env.max() * min_envelope_similarity * 0.5,
                   data_env.max())])
    # Ratio.
    ratio = stacked.min(axis=0) / stacked.max(axis=0)

    # Elimination Stage 6: Make sure the amplitudes of both don't vary too
    # much.
    if plot:
        old_time_windows = time_windows.copy()
    time_windows.mask[ratio < min_envelope_similarity] = True
    if plot:
        plt.subplot2grid(grid, (25, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="ENVELOPE AMPLITUDE SIMILARITY ELIMINATION")

    if plot:
        plt.subplot2grid(grid, (21, 0), rowspan=4)
        plt.hlines(min_envelope_similarity, xlim[0], xlim[1], color="gray",
                   linestyle="--")
        plt.text(5, min_envelope_similarity + (2) * 0.03,
                 "threshold", verticalalignment="bottom",
                 horizontalalignment="left", color="0.15",
                 path_effects=[
                 PathEffects.withStroke(linewidth=3, foreground="white")])
        plt.plot(times, ratio, color="#9B59B6",
                 label="Envelope amplitude similarity", lw=1.5)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.ylim(0.05, 1.05)
        plt.xticks([300, 600, 900, 1200, 1500, 1800])
        plt.xlim(xlim)
        plt.gca().xaxis.set_ticklabels([])
        plt.legend(loc="lower right", fancybox=True, framealpha=0.5,
                   fontsize="small")

    # First minimum window length elimination stage. This is cheap and if
    # not done it can easily destabilize the peak-and-trough marching stage
    # which would then have to deal with way more edge cases.
    if plot:
        old_time_windows = time_windows.copy()
    min_length = \
        min(minimum_period / dt * min_length_period, maximum_period / dt)
    for i in flatnotmasked_contiguous(time_windows):
        # Step 7: Throw away all windows with a length of less then
        # min_length_period the dominant period.
        if (i.stop - i.start) < min_length:
            time_windows.mask[i.start: i.stop] = True
    if plot:
        plt.subplot2grid(grid, (26, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="MINIMUM WINDOW LENGTH ELIMINATION 1")

    # -------------------------------------------------------------------------
    # Peak and trough marching algorithm
    # -------------------------------------------------------------------------
    final_windows = []
    for i in flatnotmasked_contiguous(time_windows):
        # Cut respective windows.
        window_npts = i.stop - i.start
        synthetic_window = synth[i.start: i.stop]
        data_window = data[i.start: i.stop]

        # Find extrema in the data and the synthetics.
        data_p, data_t = find_local_extrema(data_window)
        synth_p, synth_t = find_local_extrema(synthetic_window)

        window_mask = np.ones(window_npts, dtype="bool")

        closest_peaks = find_closest(data_p, synth_p)
        diffs = np.diff(closest_peaks)

        for idx in np.where(diffs == 1)[0]:
            if idx > 0:
                start = synth_p[idx - 1]
            else:
                start = 0
            if idx < (len(synth_p) - 1):
                end = synth_p[idx + 1]
            else:
                end = -1
            window_mask[start: end] = False

        closest_troughs = find_closest(data_t, synth_t)
        diffs = np.diff(closest_troughs)

        for idx in np.where(diffs == 1)[0]:
            if idx > 0:
                start = synth_t[idx - 1]
            else:
                start = 0
            if idx < (len(synth_t) - 1):
                end = synth_t[idx + 1]
            else:
                end = -1
            window_mask[start: end] = False

        window_mask = np.ma.masked_array(window_mask,
                                         mask=window_mask)

        if window_mask.mask.all():
            continue

        for j in flatnotmasked_contiguous(window_mask):
            final_windows.append((i.start + j.start, i.start + j.stop))

    if plot:
        old_time_windows = time_windows.copy()
    time_windows.mask[:] = True
    for start, stop in final_windows:
        time_windows.mask[start:stop] = False
    if plot:
        plt.subplot2grid(grid, (27, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="PEAK AND TROUGH MARCHING ELIMINATION")

    # Loop through all the time windows, remove windows not satisfying the
    # minimum number of peaks and troughs per window. Acts mainly as a
    # safety guard.
    old_time_windows = time_windows.copy()
    for i in flatnotmasked_contiguous(old_time_windows):
        synthetic_window = synth[i.start: i.stop]
        data_window = data[i.start: i.stop]
        data_p, data_t = find_local_extrema(data_window)
        synth_p, synth_t = find_local_extrema(synthetic_window)
        if np.min([len(synth_p), len(synth_t), len(data_p), len(data_t)]) < \
                min_peaks_troughs:
            time_windows.mask[i.start: i.stop] = True
    if plot:
        plt.subplot2grid(grid, (28, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="PEAK/TROUGH COUNT ELIMINATION")

    # Second minimum window length elimination stage.
    if plot:
        old_time_windows = time_windows.copy()
    min_length = \
        min(minimum_period / dt * min_length_period, maximum_period / dt)
    for i in flatnotmasked_contiguous(time_windows):
        # Step 7: Throw away all windows with a length of less then
        # min_length_period the dominant period.
        if (i.stop - i.start) < min_length:
            time_windows.mask[i.start: i.stop] = True
    if plot:
        plt.subplot2grid(grid, (29, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="MINIMUM WINDOW LENGTH ELIMINATION 2")

    # Final step, eliminating windows with little energy.
    final_windows = []
    for j in flatnotmasked_contiguous(time_windows):
        # Again assert a certain minimal length.
        if (j.stop - j.start) < min_length:
            continue

        # Compare the energy in the data window and the synthetic window.
        data_energy = (data[j.start: j.stop] ** 2).sum()
        synth_energy = (synth[j.start: j.stop] ** 2).sum()
        energies = sorted([data_energy, synth_energy])
        if energies[1] > max_energy_ratio * energies[0]:
            if verbose:
                _log_window_selection(
                    data_trace.id,
                    "Deselecting window due to energy ratio between "
                    "data and synthetics.")
            continue

        # Check that amplitudes in the data are above the noise
        if noise_absolute / data[j.start: j.stop].ptp() > \
                max_noise_window:
            if verbose:
                _log_window_selection(
                    data_trace.id,
                    "Deselecting window due having no amplitude above the "
                    "signal to noise ratio.")
        final_windows.append((j.start, j.stop))

    if plot:
        old_time_windows = time_windows.copy()
    time_windows.mask[:] = True
    for start, stop in final_windows:
        time_windows.mask[start:stop] = False

    if plot:
        plt.subplot2grid(grid, (30, 0), rowspan=1)
        _plot_mask(time_windows, old_time_windows,
                   name="LITTLE ENERGY ELIMINATION")

    if verbose:
        _log_window_selection(
            data_trace.id,
            "Done, Selected %i window(s)" % len(final_windows))

    # Final step is to convert the index value windows to actual times.
    windows = []
    for start, stop in final_windows:
        start = data_starttime + start * data_delta
        stop = data_starttime + stop * data_delta
        windows.append((start, stop))

    if plot:
        # Plot the final windows to the data axes.
        import matplotlib.transforms as mtransforms  # NOQA
        ax = data_plot
        trans = mtransforms.blended_transform_factory(ax.transData,
                                                      ax.transAxes)
        for start, stop in final_windows:
            ax.fill_between([start * data_delta, stop * data_delta], 0, 1,
                            facecolor="#CDDC39", alpha=0.5, transform=trans)

        plt.show()

    return windows


if __name__ == '__main__':
    import doctest
    doctest.testmod()
