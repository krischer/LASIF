#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Window selection algorithm.

This module aims to provide a window selection algorithm suitable for
calculating phase misfits between two seismic waveforms.

The main function is the select_windoes() function. The selection process is a
multi-stage process. Initially all time steps are considered to be valid in
the sense as being suitable for window selection. Then a number of selectors
is applied, progressively excluding more and more time steps.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
from obspy.core.util import geodetics
from obspy.taup import getTravelTimes


def find_local_extrema(data, start_index=0):
    """
    A simplistic 1D local peak (and trough) detection algorithm.

    It is only useful for smooth data and will find ALL local extrema. A local
    minimum is defined as having larger values as its immediate neighbours. A
    local maximum is consequently defined as having smaller values in its
    immediate vicinity.

    Will also find local extrema at the domain borders.

    :param data: A 1D array over which the search will be performed.
    :type data: np.ndarray
    :param start_index: The minimum index at which extrema will be detected.
    :return: Returns a tuple with three arrays:
        * [0]: The indices of all found peaks
        * [1]: The indices of all found troughs
        * [2]: The indices of all found extreme points (peaks and troughs)

    >>> data = np.array([0, 1, 2, 1, 0, 1, 2, 1])
    >>> peaks, troughs, extrema = find_local_extrema(data)
    >>> peaks
    array([2, 6])
    >>> troughs
    array([0, 4, 7])
    >>> extrema
    array([0, 2, 4, 6, 7])
    """
    # Detect peaks.
    peaks = np.r_[True, data[1:] > data[:-1]] & \
        np.r_[data[:-1] > data[1:], True]
    peaks = np.where(peaks)[0]
    peaks = peaks[np.where(peaks >= start_index)[0]]

    troughs = np.r_[True, data[1:] < data[:-1]] & \
        np.r_[data[:-1] < data[1:], True]
    troughs = np.where(troughs)[0]
    troughs = troughs[np.where(troughs >= start_index)[0]]

    # Now create a merged version.
    if np.intersect1d(peaks, troughs, assume_unique=True):
        msg = "Error. Peak and troughs should not be identical! Something " \
            "went wrong. Please fix it or contact the developers."
        raise Exception(msg)
    extrema = np.concatenate([peaks, troughs])
    extrema.sort()

    return peaks, troughs, extrema


def find_closest(ref_array, target):
    """
    For every value in target, find the index of ref_array to which
    the value is closest.

    from http://stackoverflow.com/a/8929827/1657047

    :param ref_array: The reference array. Must be sorted!
    :type ref_array: np.ndarray
    :param target: The target array.
    :type target: np.ndarray

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


def select_windows(data_trace, synthetic_trace, ev_lat, ev_lng,
        ev_depth_in_km, st_lat, st_lng, minimum_period, maximum_period):
    """
    Window selection algorithm for picking windows suitable for misfit
    calculation based on phase differences.

    :param data_trace:
    :param synthetic_trace:
    :param ev_lat:
    :param ev_lng:
    :param ev_depth_in_km:
    :param st_lat:
    :param st_lng:
    :param minimum_period:
    :param maximum_period:
    """
    dt = synthetic_trace.stats.delta
    npts = synthetic_trace.stats.npts
    dist_in_deg = geodetics.locations2degrees(st_lat, st_lng, ev_lat, ev_lng)
    dist_in_km = geodetics.calcVincentyInverse(st_lat, st_lng, ev_lat,
        ev_lng)[0] / 1000.0
    tts = getTravelTimes(dist_in_deg, ev_depth_in_km, model="ak135")
    first_tt_arrival = min([_i["time"] for _i in tts])

    # The window length. Currently set to one dominant period of the
    # synthetics. Make sure it is an uneven number; just to have an easy
    # midpoint definition.
    window_length = int(round(float(minimum_period) / dt))

    if not window_length % 2:
        window_length += 1

    # Naive sliding window approach. Can be replaced by more efficient
    # variants if necessary.

    # Allocate arrays to collect the time dependent values.
    sliding_time_shift = np.ma.masked_all(npts, dtype="float32")
    max_cc_coeff = np.ma.masked_all(npts, dtype="float32")

    taper = np.hanning(window_length)

    for start_idx, end_idx, midpoint_idx in _window_generator(npts,
            window_length):
        # Slice windows. Create a copy to be able to taper without affecting
        #  the original time series.
        data_window = data_trace.data[start_idx: end_idx].copy() * taper
        synthetic_window = synthetic_trace.data[start_idx: end_idx].copy() * \
            taper

        # Skip windows that have essentially no energy to avoid instabilities.
        if synthetic_window.ptp() < synthetic_trace.data.ptp() * 0.001:
            continue

        # Calculate the time shift. Here this is defined as the shift of the
        # synthetics relative to the data. So a value of 2 means that the
        # synthetics are 2 timesteps later then the data.
        cc = np.correlate(data_window, synthetic_window, mode="full")

        time_shift = cc.argmax() - window_length + 1
        # Express the time shift in fraction of dominant period.
        sliding_time_shift[midpoint_idx] = (time_shift * dt) / \
            minimum_period

        # Normalized cross correlation.
        max_cc_value = cc.max() / np.sqrt((synthetic_window ** 2).sum() *
            (data_window ** 2).sum())
        max_cc_coeff[midpoint_idx] = max_cc_value

    threshold_travel_time = 2.0

    # Step 1
    time_windows = np.ma.ones(npts)
    time_windows.mask = np.zeros(npts)

    # Step 2: Mark everything more then half a dominant period before the
    # first theoretical arrival as positive.
    time_windows.mask[:int(np.ceil((first_tt_arrival - minimum_period * 0.5)
        / dt))] = True

    # Step 3: Mark everything more then half a dominant period after the
    # chosen threshold surface wave velocity time as negative
    time_windows.mask[int(np.floor(dist_in_km /
        threshold_travel_time / dt)):] = True

    # Step 4: Mark everything with an absolute travel time shift of more then
    # 0.2 time the dominant period as negative
    time_windows.mask[np.abs(sliding_time_shift) > 0.2] = True

    # Step 5: Mark the area around every "travel time shift jump" (based on
    # the traveltime time difference) negative. The width of the area is
    # currently chosen to be a tenth of a dominant period to each side.
    sample_buffer = int(np.ceil(minimum_period / dt * 0.1))
    indices = np.ma.where(np.abs(np.diff(sliding_time_shift)) > 0.1)[0]
    for index in indices:
        time_windows.mask[index - sample_buffer: index + sample_buffer] = \
            True

    # Step 6: Mark all areas where the normalized cross correlation coefficient
    # is under 0.7 as negative
    time_windows.mask[max_cc_coeff < 0.7] = True

    # Step 7: Throw away all windows with a length of less then 0.5 the
    # dominant period.
    min_length = min(minimum_period / dt * 1.5, maximum_period / dt)
    final_windows = []
    for i in np.ma.flatnotmasked_contiguous(time_windows):
        # Assert a certain minimal length.
        if (i.stop - i.start) < min_length:
            continue

        window_npts = i.stop - i.start
        synthetic_window = synthetic_trace.data[i.start: i.stop]
        data_window = data_trace.data[i.start: i.stop]

        # Get the local extreme value of the data as well as the synthetics.
        data_p, data_t, data_extrema = find_local_extrema(data_window, 0)
        synth_p, synth_t, synth_extrema = find_local_extrema(synthetic_window,
            0)

        if 0 in [len(synth_p), len(synth_t), len(data_p), len(data_t)]:
            continue

        if 1 in [len(synth_p), len(synth_t), len(data_p), len(data_t)]:
            continue

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

        window_mask = np.ma.masked_array(window_mask, mask=window_mask)
        if window_mask.mask.all():
            continue
        for j in np.ma.flatnotmasked_contiguous(window_mask):
            # Again assert a certain minimal length.
            if (j.stop - j.start) < min_length:
                continue
            # Now compare the energy in the data window in the synthetic
            # window. If they differ by more then one order of magnitude,
            # discard them.
            data_energy = (data_window[j.start: j.stop] ** 2).sum()
            synth_energy = (synthetic_window[j.start: j.stop] ** 2).sum()
            energies = sorted([data_energy, synth_energy])
            if energies[1] > 10.0 * energies[0]:
                continue
            final_windows.append((i.start + j.start, i.start + j.stop))

    return final_windows


def plot_windows(data_trace, synthetic_trace, windows, dominant_period,
        filename=None, debug=False):
    """
    Helper function plotting the picked windows in some variants. Useful for
    debugging and checking what's actually going on.

    If using the debug option, please use the same data_trace and
    synthetic_trace as you used for the select_windows() function. They will
    be augmented with certain values used for the debugging plots.

    :param data_trace: The data trace.
    :type data_trace: obspy.core.trace.Trace
    :param synthetic_trace: The synthetic trace.
    :type synthetic_trace: obspy.core.trace.Trace
    :param windows: The windows, as returned by select_windows()
    :type windows: list
    :param dominant_period: The dominant period of the data. Used for the
        tapering.
    :type dominant_period: float
    :param filename: If given, a file will be written. Otherwise the plot
        will be shown.
    :type filename: basestring
    :param debug: Toggle plotting debugging information. Optional. Defaults
        to False.
    :type debug: bool
    """
    import matplotlib.pylab as plt
    from obspy.signal.invsim import cosTaper

    plt.figure(figsize=(16, 10))
    plt.subplots_adjust(hspace=0.3)

    npts = synthetic_trace.stats.npts

    # Plot the raw data.
    time_array = np.linspace(0, (npts - 1) * synthetic_trace.stats.delta, npts)
    plt.subplot(411)
    plt.plot(time_array, data_trace.data, color="black", label="data")
    plt.plot(time_array, synthetic_trace.data, color="red",
        label="synthetics")
    plt.xlim(0, time_array[-1])
    plt.title("Raw data")

    # Plot the chosen windows.
    bottom = np.ones(npts) * -10.0
    top = np.ones(npts) * 10.0
    for left_idx, right_idx in windows:
        top[left_idx: right_idx + 1] = -10.0
    plt.subplot(412)
    plt.plot(time_array, data_trace.data, color="black", label="data")
    plt.plot(time_array, synthetic_trace.data, color="red",
             label="synthetics")
    ymin, ymax = plt.ylim()
    plt.fill_between(time_array, bottom, top, color="red", alpha="0.5")
    plt.xlim(0, time_array[-1])
    plt.ylim(ymin, ymax)
    plt.title("Chosen windows")

    # Plot the tapered data.
    final_data = np.zeros(npts)
    final_data_scaled = np.zeros(npts)
    synth_data = np.zeros(npts)
    synth_data_scaled = np.zeros(npts)

    for left_idx, right_idx in windows:
        right_idx += 1
        length = right_idx - left_idx

        # Setup the taper.
        p = (dominant_period / synthetic_trace.stats.delta / length) / 2.0
        if p >= 0.5:
            p = 0.49
        elif p < 0.1:
            p = 0.1
        taper = cosTaper(length, p=p)

        data_window = taper * data_trace.data[left_idx: right_idx].copy()
        synth_window = taper * synthetic_trace.data[left_idx: right_idx].copy()

        data_window_scaled = data_window / data_window.ptp() * 2.0
        synth_window_scaled = synth_window / synth_window.ptp() * 2.0

        final_data[left_idx: right_idx] = data_window
        synth_data[left_idx: right_idx] = synth_window
        final_data_scaled[left_idx: right_idx] = data_window_scaled
        synth_data_scaled[left_idx: right_idx] = synth_window_scaled

    plt.subplot(413)
    plt.plot(time_array, final_data, color="black")
    plt.plot(time_array, synth_data, color="red")
    plt.xlim(0, time_array[-1])
    plt.title("Tapered windows")

    plt.subplot(414)
    plt.plot(time_array, final_data_scaled, color="black")
    plt.plot(time_array, synth_data_scaled, color="red")
    plt.xlim(0, time_array[-1])
    plt.title("Tapered windows, scaled to same amplitude")

    if debug:
        first_valid_index = data_trace.stats.first_valid_index * \
            synthetic_trace.stats.delta
        noise_level = data_trace.stats.noise_level

        data_p, data_t, data_e = find_local_extrema(data_trace.data,
            start_index=first_valid_index)
        synth_p, synth_t, synth_e = find_local_extrema(synthetic_trace.data,
            start_index=first_valid_index)

        for _i in xrange(1, 3):
            plt.subplot(4, 1, _i)
            ymin, ymax = plt.ylim()
            xmin, xmax = plt.xlim()
            plt.vlines(first_valid_index, ymin, ymax, color="green",
                label="Theoretical First Arrival")
            plt.hlines(noise_level, xmin, xmax, color="0.5",
                       label="Noise Level", linestyles="--")
            plt.hlines(-noise_level, xmin, xmax, color="0.5", linestyles="--")

            plt.hlines(noise_level * 5, xmin, xmax, color="0.8",
                       label="Minimal acceptable amplitude", linestyles="--")
            plt.hlines(-noise_level * 5, xmin, xmax, color="0.8",
                       linestyles="--")
            if _i == 2:
                plt.scatter(time_array[data_e], data_trace.data[data_e],
                            color="black", s=10)
                plt.scatter(time_array[synth_e], synthetic_trace.data[synth_e],
                            color="red", s=10)
            plt.ylim(ymin, ymax)
            plt.xlim(xmin, xmax)

        plt.subplot(411)
        plt.legend(prop={"size": "small"})

    plt.suptitle(data_trace.id)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()