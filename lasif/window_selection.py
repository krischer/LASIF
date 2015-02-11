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
    :type data: :class:`numpy.ndarray`
    :param start_index: The minimum index at which extrema will be detected.
    :return: Returns a tuple with three arrays:
        * [0]: The indices of all found peaks
        * [1]: The indices of all found troughs
        * [2]: The indices of all found extreme points (peaks and troughs)

    >>> import numpy as np
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


def select_windows(data_trace, synthetic_trace, event_latitude,
                   event_longitude, event_depth_in_km,
                   station_latitude, station_longitude, minimum_period,
                   maximum_period,
                   min_cc=0.10, max_noise=0.10, max_noise_window=0.4,
                   min_velocity=2.4, threshold_shift=0.30,
                   threshold_correlation=0.75, min_length_period=1.5,
                   min_peaks_troughs=2, max_energy_ratio=2.0, quiet=True):
    """
    Window selection algorithm for picking windows suitable for misfit
    calculation based on phase differences.

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
    :param threshold_correlation: Minimum normalised correlation coeficient
        within a window.
    :type threshold_correlation: float
    :param min_length_period: Minimum length of the time windows relative to
        the minimum period.
    :type min_length_period: float
    :param min_peaks_troughs: Minimum number of extrema in an individual
        time window (excluding the edges).
    :type min_peaks_troughs: float
    :param max_energy_ratio: Maximum energy ratio between data and
        synthetics within a time window.
    :type max_energy_ratio: float
    :param quiet: Be quiet and don't print anything.
    :type quiet: bool
    """
    if not quiet:
        print "* ---------------------------"
        print "* autoselect " + data_trace.id

    data_starttime = data_trace.stats.starttime
    data_delta = data_trace.stats.delta

    # =========================================================================
    # initialisations
    # =========================================================================

    dt = synthetic_trace.stats.delta
    npts = synthetic_trace.stats.npts
    dist_in_deg = geodetics.locations2degrees(station_latitude,
                                              station_longitude,
                                              event_latitude, event_longitude)
    dist_in_km = geodetics.calcVincentyInverse(
        station_latitude, station_longitude, event_latitude,
        event_longitude)[0] / 1000.0
    tts = getTravelTimes(dist_in_deg, event_depth_in_km, model="ak135")
    first_tt_arrival = min([_i["time"] for _i in tts])

    # Number of samples in the sliding window. Currently, the length of the
    # window is set to a multiple of the dominant period of the synthetics.
    # Make sure it is an uneven number; just to have a trivial midpoint
    # definition.
    window_length = int(round(float(2 * minimum_period) / dt))

    if not window_length % 2:
        window_length += 1

    # Allocate arrays to collect the time dependent values.
    sliding_time_shift = np.zeros(npts, dtype="float32")
    max_cc_coeff = np.zeros(npts, dtype="float32")

    taper = np.hanning(window_length)

    # =========================================================================
    # check if whole seismograms are sufficiently correlated and estimate
    # noise level
    # =========================================================================

    synth = synthetic_trace.data
    data = data_trace.data

    #  compute correlation coefficient
    norm = np.sqrt(np.sum(data ** 2)) * np.sqrt(np.sum(synth ** 2))
    cc = np.sum(data * synth) / norm
    if not quiet:
        print "** correlation coefficient: " + str(cc)

    #  estimate noise level from waveforms prior to the first arrival
    idx_end = int(np.ceil((first_tt_arrival - 0.5 * minimum_period) / dt))
    idx_end = max(10, idx_end)
    idx_start = int(np.ceil((first_tt_arrival - 2.5 * minimum_period) / dt))
    idx_start = max(10, idx_start)

    if idx_start >= idx_end:
        idx_start = max(0, idx_end - 10)

    noise_absolute = data[idx_start:idx_end].ptp()
    noise_relative = noise_absolute / data.ptp()

    if not quiet:
        print "** absolute noise level: " + str(noise_absolute) + " m/s"
        print "** relative noise level: " + str(noise_relative)

    #  rejection criteria
    accept = True

    if (cc < min_cc) and (noise_relative > max_noise / 3.0):
        if not quiet:
            print "** correlation " + str(cc) + \
                " is below threshold value of " + str(min_cc)
        accept = False

    if noise_relative > max_noise:
        if not quiet:
            print "** noise level " + str(noise_relative) + \
                " is above threshold value of " + str(max_noise)
        accept = False

    if accept is False:
        if not quiet:
            print "* autoselect done, 0 windows selected"
        return []

    # =========================================================================
    # compute sliding time shifts and correlation coefficients
    # =========================================================================

    for start_idx, end_idx, midpoint_idx in _window_generator(npts,
                                                              window_length):

        # Slice windows. Create a copy to be able to taper without affecting
        # the original time series.
        data_window = data_trace.data[start_idx: end_idx].copy() * taper
        synthetic_window = \
            synthetic_trace.data[start_idx: end_idx].copy() * taper

        # Skip windows that have essentially no energy to avoid instabilities.
        if synthetic_window.ptp() < synthetic_trace.data.ptp() * 0.001:
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

    # =========================================================================
    # compute the initial mask, i.e. intervals (windows) where no measurements
    # are made.
    # =========================================================================

    # Step 1: Initialise masked arrays. The mask will be set to True where no
    # windows are chosen.
    time_windows = np.ma.ones(npts)
    time_windows.mask = np.zeros(npts)

    # Step 2: Mark everything more then half a dominant period before the first
    # theoretical arrival as positive.
    time_windows.mask[:int(np.ceil(
                      (first_tt_arrival - minimum_period * 0.5) / dt))] = True

    # Step 3: Mark everything more then half a dominant period after the
    # threshold arrival time - computed from the threshold velocity - as
    # negative.
    time_windows.mask[int(np.floor(dist_in_km / min_velocity / dt)):] = True

    # Step 4: Mark everything with an absolute travel time shift of more than
    # threshold_shift times the dominant period as negative
    time_windows.mask[np.abs(sliding_time_shift) > threshold_shift] = True

    # Step 5: Mark the area around every "travel time shift jump" (based on
    # the traveltime time difference) negative. The width of the area is
    # currently chosen to be a tenth of a dominant period to each side.
    sample_buffer = int(np.ceil(minimum_period / dt * 0.1))
    indices = np.ma.where(np.abs(np.diff(sliding_time_shift)) > 0.1)[0]
    for index in indices:
        time_windows.mask[index - sample_buffer: index + sample_buffer] = True

    # Step 6: Mark all areas where the normalized cross correlation coefficient
    # is under threshold_correlation as negative
    time_windows.mask[max_cc_coeff < threshold_correlation] = True

    # =========================================================================
    #  Make the final window selection.
    # =========================================================================

    min_length = min(minimum_period / dt * min_length_period,
                     maximum_period / dt)
    final_windows = []

    if np.ma.flatnotmasked_contiguous(time_windows) is None:
        windows = []
        return

    #  loop through all the time windows
    for i in np.ma.flatnotmasked_contiguous(time_windows):

        # Step 7: Throw away all windows with a length of less then
        # min_length_period the dominant period.
        if (i.stop - i.start) < min_length:
            continue

        window_npts = i.stop - i.start
        synthetic_window = synthetic_trace.data[i.start: i.stop]
        data_window = data_trace.data[i.start: i.stop]

        # Step 8: Exclude windows without a real peak or trough (except for the
        # edges).
        data_p, data_t, data_extrema = find_local_extrema(data_window, 0)
        synth_p, synth_t, synth_extrema = find_local_extrema(synthetic_window,
                                                             0)
        if np.min([len(synth_p), len(synth_t), len(data_p), len(data_t)]) < \
                min_peaks_troughs:
            continue

        # Step 9: Peak and trough matching algorithm
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

        # Step 10: Check if the time windows have sufficiently similar energy
        # and are above the noise
        for j in np.ma.flatnotmasked_contiguous(window_mask):

            # Again assert a certain minimal length.
            if (j.stop - j.start) < min_length:
                continue

            # Compare the energy in the data window and the synthetic window.
            data_energy = (data_window[j.start: j.stop] ** 2).sum()
            synth_energy = (synthetic_window[j.start: j.stop] ** 2).sum()
            energies = sorted([data_energy, synth_energy])
            if energies[1] > max_energy_ratio * energies[0]:
                continue

            # Check that amplitudes in the data are above the noise
            if noise_absolute / data_window[j.start: j.stop].ptp() > \
                    max_noise_window:
                continue

            final_windows.append((i.start + j.start, i.start + j.stop))

    if not quiet:
        print "* autoselect done, " + str(len(final_windows)) + \
            " window(s) selected"

    # Final step is to convert the index value windows to actual times.
    windows = []
    for start, stop in final_windows:
        start = data_starttime + start * data_delta
        stop = data_starttime + stop * data_delta
        windows.append((start, stop))

    return windows


if __name__ == '__main__':
    import doctest
    doctest.testmod()
