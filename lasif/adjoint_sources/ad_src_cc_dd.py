#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
An implementation of the Double-difference adjoint sources introduced
by Yuan et al 2016.

:copyright:
    Solvi Thrastarson (soelvi.thrastarson@erdw.ethz.ch)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
from scipy.integrate import simps
from obspy.signal.invsim import cosine_taper
import obspy.signal.cross_correlation as crosscorr
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics.base import kilometer2degrees
from lasif import LASIFError
from obspy.core import Trace


def find_comparable_stations(event_name, station_info, window_set, comm):
    """
    Find stations which are of comparable distances from the station where
    the adjoint source is calculated.
    A criterion for this is yet to be set in stone but for now we can just
    use a minimum/maximum distance criterion.
    :param event_name: Name of the event which caused the measured data
    :param station_info: Name of the station which the adjoint source is
    calculated at
    :param window_set: Used to figure out whether trace has windows.
    :param comm: Used to access other parts of code
    :return: used_stations: A list containing station names that will be used
    """
    # Get stations for event, preferably not all of them
    # Calculate distance from reference station
    # If it fits the criterion, calculate great circle from event
    # Organize information into a dictionary with station name as key.
    # For now we just use stations in similar latitude longitude range.

    ############ CHECK IF STATION HAS A WINDOW ON THIS COMPONENT ##########

    stations = comm.query.get_all_stations_for_event(event_name)
    lat = station_info["latitude"]
    lon = station_info["longitude"]
    used_stations = list()

    for key in stations:
        sta_lat = stations[key]["latitude"]
        sta_lon = stations[key]["longitude"]
        if not np.abs(sta_lat - lat) > 4 and not np.abs(sta_lon - lon) > 8 \
                and not sta_lat == lat and not sta_lon == lon:

            windows_group = comm.windows.get(window_set)
            if len(windows_group.get_all_windows_for_station(key)) != 0:
                used_stations.append(key)

            continue

    return used_stations


def time_shift_trace(trace, dt, time_shift):
    """
    Time shift a certain trace for an amount of seconds.
    :param trace: Trace to be time shifted
    :param dt: Sampling interval of the trace
    :param time_shift: Seconds to be timeshifted. Positive shifts to the right.
    :return: Time shifted trace.
    """
    # If trace is in further ahead than the one it needs to be compared to
    # Values are taken from the back and put into the front.
    # This should work well with windowed and zero padded traces

    if np.abs(time_shift) < dt:  # We need to shift signal for at least a dt
        return trace

    steps = int(-time_shift / dt)
    if time_shift < 0.0:
        # If time shift is negative, the trace is behind and needs to be
        # rolled forward.
        time_shifted_trace = np.roll(trace, steps)
    else:
        # If time shift is positive, the trace is in front and needs to shift
        # back the roll method does not support negative numbers and we thus
        # need to trick it with a couple of time reversals.
        time_shifted_trace = np.roll(trace[::-1], -steps)[::-1]

    return time_shifted_trace


def shift_window(window, comm, station, event,
                 station_loc, start_time):
    """
    In order to be able to properly compare the windows they need to be shifted
    in time for the compared station. The appropriate time shift is
    approximated using the ratio of great circle distances between the two
    source receiver pairs and comparing it to the start time of the
    initial window.
    This method proved more stable than using cross_correlation to estimate
    shift.
    :param window: Time stamp of window used to calculate adjoint source
    :param comm: Communicator to access rest of code
    :param station: Name of station that will get its window shifted
    :param event: Name of event, used to find its coordinates
    :param station_loc: Coordinates of reference station.
    :param start_time: start_time of original trace. Basically event origin.
    :return: new_window: New window to use for the compared trace.
    :return: time: Time in seconds that the window is shifted.
    """

    station = comm.query.get_coordinates_for_station(event, station)
    ev_lat = comm.events.get(event)["latitude"]
    ev_lon = comm.events.get(event)["longitude"]
    ref = gps2dist_azimuth(ev_lat, ev_lon,
                           station_loc["latitude"], station_loc["longitude"])

    comp = gps2dist_azimuth(ev_lat, ev_lon,
                            station["latitude"], station["longitude"])

    ratio = kilometer2degrees(comp[0] / 1000) / kilometer2degrees(ref[0] / 1000)

    if ratio == 0.0:
        raise LASIFError("Distance ratio is 0.0 can not compute")
    time = (window[0] - start_time) * ratio

    # Shift initial window according to time shift
    window_start = start_time + time
    window_end = window_start + (window[1] - window[0])
    new_window = [window_start, window_end]
    end_time = comm.project.solver_settings["end_time"]
    end_time = start_time + end_time

    if new_window[0] <= start_time or new_window[1] >= end_time:
        new_window = [0, 0]
    time = window_start - window[0]

    return time, new_window


def window_seismogram(trace, window, original_stats):
    """
    Get seismogram for station event pair. Make a zero padded window using
    the window time stamps. Pad the window as well.
    :param trace: Trace to be windowed
    :param window: Its window time stamps
    :param original_stats: An obspy stats component containing time info
    :return: windowed_trace: zero padded window into the trace.
    """

    trace = Trace(trace, header=original_stats)
    windowed = trace.trim(starttime=window[0], endtime=window[1])
    windowed_trace = windowed.trim(original_stats.starttime,
                                   original_stats.endtime,
                                   pad=True, fill_value=0.0).data

    return windowed_trace


def cc_time_shift(reference, compared, dt, shift):
    """
    Compute the time shift between two traces by cross-correlating them.
    :param reference: reference trace
    :param compared: trace compared to reference trace
    :param dt: Time step
    :param shift: How far to shift to both directions in the cross-correlation
    :return: A value of the time misfit between the two traces.
    """
    # See whether the lengths are the same:
    if not len(reference) == len(compared):
        raise LASIFError("\n\n Data and Synthetics do not have equal number"
                         " of points. Might be something wrong with your"
                         " processing.")

    cc = crosscorr.correlate(a=np.array(reference), b=np.array(compared),
                             shift=shift)

    shift = np.argmax(cc) - shift + 1  # Correct shift in the cross_corr
    time_shift = dt * shift

    return time_shift


def taper(ad_src, min_period, dt):
    """
    Taper the adjoint source window using a cosine taper
    :param ad_src: the adjoint source trace
    :param min_period: Minimum period used for finding tapering window
    :param dt: Time step
    :return: ad_src_tapered: A tapered version of the adjoint source.
    """

    orig_length = len(ad_src)

    n_zeros = np.nonzero(ad_src)  # Find nonzero indices
    ad_src = np.trim_zeros(ad_src)  # Trim off padded zeros
    len_window = len(ad_src) * dt  # Use that to find

    # Taper the adjoint source
    ratio = min_period * 2.0 / len_window
    p = ratio / 2.0  # We want the minimum window to taper 25% off each side
    if p > 1.0:  # For manually picked small windows. Should not happen.
        p = 1.0
    window = cosine_taper(len(ad_src), p=p)
    ad_src = ad_src * window
    front_pad = np.zeros(n_zeros[0][0])
    back_pad = np.zeros(orig_length - n_zeros[0][-1] - 1)
    ad_src_tapered = np.concatenate([front_pad, ad_src, back_pad])

    return ad_src_tapered


def window_too_short(window, min_period):
    """
    Window needs to be of length twice minimum period to be used.
    For automatic windows, this should never be a problem.
    :param window: time stamp of window
    :param min_period: minimum modelled period
    :return: binary tag giving an indication of window quality
    """

    if window[1] - window[0] < 2 * min_period:
        return True
    else:
        return False


def adsrc_cc_time_shift(dt, data, synthetics, min_period):
    """
       :rtype: dictionary
       :returns: Return a dictionary with three keys:
           * adjoint_source: The calculated adjoint source as a numpy array
           * misfit: The misfit value
           * messages: A list of strings giving additional hints to what
                happened in the calculation.
    """

    # Move the minimum period in each direction to avoid cycle skip.
    shift = int(min_period / dt)  # This can be adjusted if you wish so.
    # Compute time shift between the two traces.
    time_shift = cc_time_shift(data, synthetics, dt, shift)
    misfit = 1.0 / 2.0 * time_shift ** 2

    # Now we have the time shift. We need velocity of synthetics.
    vel_syn = np.gradient(synthetics) / dt
    norm_const = simps(np.square(vel_syn), dx=dt)
    ad_src = (time_shift / norm_const) * vel_syn * dt

    # Taper it and time reverse it
    ad_src = taper(ad_src, min_period, dt)
    ad_src = ad_src[::-1]

    return ad_src, misfit


def adsrc_cc_dd(dt, data, synthetic, comp_data, comp_synthetics, min_period,
                time):
    """
    Calculate adjoint source for a single station pair. Source is tapered.
    This adjoint source will later be superimposed on top of other adjoint
    sources.
    We take two data traces, calculate the time shift between them using
    cc_time_shift.
    Do the same thing with the synthetics.
    Difference between these two time shifts is the misfit value.
    Use (Yuan 2016) formulation to calculate adjoint source.
    Taper adjoint source using taper
    Return a trace containing this adjoint source and the misfit value.
    :param dt: time step
    :param data: Windowed observed data
    :param synthetic: Windowed synthetics
    :param comp_data: Recordings at compared station. Windowed
    :param comp_synthetics: Synthetics at compared station. Windowed
    :param min_period: Minimum period... used for tapering
    :param time: The time shift between the initial picked windows.
    :return: ad_src: Adjoint source trace for this pair
    """
    # Take two data traces, calculate time shift between them
    # using cc_time_shift.
    # Do the same thing with the synthetics.
    # Difference between them is the misfit value.
    # Use (Yuan 2016) formulation to calculate adjoint source.
    # Taper the adjoint source using adsrc_taper
    # Return a trace containing this adjoint source.

    shift = int(len(data) / 4.0 / dt)  # Again an arbitrary number

    data_shift = cc_time_shift(reference=data, compared=comp_data, dt=dt,
                               shift=shift)
    synthetic_shift = cc_time_shift(reference=synthetic,
                                    compared=comp_synthetics, dt=dt,
                                    shift=shift)

    dd_difference = synthetic_shift - data_shift
    misfit = np.square(dd_difference)

    # Not fully sure how to implement the normalization constant tbh
    # Still have to timeshift the synthetic at rec_j
    synthetic_time_shift = time_shift_trace(comp_synthetics, dt,
                                            time)

    acc_rec_i = np.gradient(np.gradient(synthetic_time_shift) / dt) / dt
    acc_rec_j = np.gradient(np.gradient(comp_synthetics) / dt) / dt
    norm_constant = simps(acc_rec_i * comp_synthetics + synthetic_time_shift
                          * acc_rec_j)

    vel_syn_rec_j = np.gradient(comp_synthetics) / dt
    vel_syn_rec_j_time_shifted = time_shift_trace(vel_syn_rec_j, dt,
                                                  time)
    ad_src = dd_difference / norm_constant * vel_syn_rec_j_time_shifted

    # Taper
    ad_src = taper(ad_src, min_period=min_period, dt=dt)
    # Possible ploting:
    # import matplotlib.pyplot as plt
    # plt.plot(t, comp_synthetics / np.max(comp_synthetics), color="green",
    #          label="Compared_orig")
    # plt.plot(t, synthetic_time_shift / np.max(synthetic_time_shift),
    #          color="red", label="Compared_shifted")
    # plt.plot(t, synthetic / np.max(synthetic), color="blue",
    #           label="reference")
    # plt.plot(t, ad_src / np.max(ad_src), color="black",
    #          label="adjoint", linestyle="--")
    # plt.legend()
    # plt.show()

    # And finally time reverse it:
    ad_src = ad_src[::-1]

    return ad_src, misfit


def combine_adsrc(ad_src_orig, ad_src_comb):
    """
    Take an adjoint source and superimpose another one on top of it.
    :param ad_src_orig: Original adjoint source
    :param ad_src_comb: The adjoint source to be superimposed on top
    :return: ad_src: The combined adjoint source.
    """
    ad_src = ad_src_orig + ad_src_comb

    return ad_src


def double_difference_adjoint(t, data, synthetic, window, min_period, event,
                              station_name, original_stats, iteration, comm,
                              window_set, plot=False):
    """
    A function to calculate the double difference adjoint sources.
    The input is the synthetics and the data along with window parameters.
    The algorithm finds comparable stations via find_comparable_stations,
    saves these stations into a list.
    If the list is empty, it will spit out a cross_correlation time shift
    adjoint source.

    If list is not empty, the algorithm loops through them, uses a ratio of
    great circle distances from source to receiver to estimate a shift of
    the window. It then windows the synthetics and data at the compared
    stations and compares the time shift between data and the time shift
    between synthetics between the two stations. This is then used to
    construct an adjoint source.
    :param t: time of each sample of data and synthetics
    :param data: Observed data
    :param synthetic: Synthetics
    :param window: Time stamp of window, faster if all windows for trace
    :param min_period: Minimum period used for tapering
    :param event: Name of the event.
    :param station_name: Name of reference station
    :param original_stats: stats of original trace to keep track
    :param iteration: Name of current iteration to find synthetics
    :param comm: Communicator to access other parts of code.
    :param window_set: Name of used window set.
    :param plot: Binary to see whether user wants to plot adjoint source or not
    :return: ad_src: double-difference adjoint source.
    """

    if window_too_short(window, min_period):
        misfit = 0.0
        ad_src = np.zeros(len(synthetic))
        messages = list()
        messages.append("Window too short")
        ret_dict = {"adjoint_source": ad_src,
                    "misfit_value": misfit,
                    "details": {"messages": messages}}
        return ret_dict

    dt = t[1] - t[0]
    start_time = original_stats.starttime
    iteration = comm.iterations.get_long_iteration_name(iteration)
    filename = comm.waveforms.get_asdf_filename(
        event, data_type="processed", tag_or_iteration=
        comm.waveforms.preprocessing_tag)

    station_loc = comm.query.get_coordinates_for_station(event, station_name)
    win_synth = window_seismogram(synthetic, window, original_stats)
    win_data = window_seismogram(data, window, original_stats)


    # adsrc_cc_timeshift
    ad_src, misfit = adsrc_cc_time_shift(dt, win_data, win_synth, min_period)

    # find_comparable_stations
    stations = find_comparable_stations(event, station_loc, window_set, comm)

    if len(stations) == 0:
        print("No stations to compare with")
    else:
        print(f"I have {len(stations)} amount of comparable stations!")

    print(stations)
    for station in stations:
        waves = comm.query.get_matching_waveforms(
                event=event, iteration=iteration,
                station_or_channel_id=station)
        data = waves.data[0].data
        synthetic = waves.synthetics[0].data

        time, window_sta = shift_window(window, comm, station, event,
                                        station_loc, start_time)
        if window_sta == [0, 0]:
            print(f'Window shifting for station {station} does not work.')
            continue

        synthetic = window_seismogram(synthetic, window_sta, original_stats)
        synthetic = taper(synthetic, min_period, dt)

        data = window_seismogram(data, window_sta, original_stats)
        data = taper(data, min_period, dt)

        new_ad_src, dd_misfit = adsrc_cc_dd(dt, data=win_data,
                                            synthetic=win_synth,
                                            comp_data=data,
                                            comp_synthetics=synthetic,
                                            min_period=min_period,
                                            time=time)
        ad_src = combine_adsrc(ad_src, new_ad_src)
        misfit += np.square(dd_misfit / 2.0)

    misfit = 1.0 / 2.0 * misfit
    messages = list()
    messages.append(f"len(stations) amount of stations used.")

    ret_dict = {"adjoint_source": ad_src,
                "misfit_value": misfit,
                "details": {"messages": messages}}
    if plot:
        adjoint_source_plot(t, win_data, win_synth, ad_src, misfit, len(stations))

    return ret_dict


def adjoint_source_plot(t, data, synthetic, adjoint_source, misfit, stations):

    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(t, data, color="0.2", label="Data", lw=2)
    plt.plot(t, synthetic, color="#bb474f",
             label="Synthetic", lw=2)

    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.subplot(212)
    plt.plot(t, adjoint_source[::-1], color="#2f8d5b", lw=2,
             label="Adjoint Source")
    plt.grid()
    plt.legend(fancybox=True, framealpha=0.5)

    plt.title(f"DD_Ad_src with a Misfit of {misfit}. "
              f"Stations compared: {stations}")
