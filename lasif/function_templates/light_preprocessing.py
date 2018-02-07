#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Function to help with handling huge data sets. It takes the downloaded data
and immediately processes it to a certain extent without ruining the data.
One has to be very careful when using this function because it is made
specifically to replace the raw data with processed data.
"""
from scipy import signal
import os
from pyasdf import ASDFDataSet


def light_preprocessing_function(event, light_process_parameters=None):
    """
    Take downloaded data from event. lowpass filter it and downsample.
    It will be downsampled in a way that the resulting traces are still
    with a maximum frequencies which are at least an order of magnitude
    higher than the maximum frequencies that will be used in the actual
    inversion process.
    :param event: The event where data should be lightly processed
    :param light_process_parameters: A dictionary with information
        on how to process the data
    :return:
    """

    if not light_process_parameters:

        light_process_parameters = {}

        light_process_parameters["dt"] = 0.25
        light_process_parameters["max_freq"] = 1.0
        event_file_name = event.get_file_name() # Do better
        light_process_parameters["event_file_name"] = event_file_name

    ds = ASDFDataSet(light_process_parameters["event_file_name"],
                     compression=None)

    def light_process_function(st,inv):
        def zerophase_chebychev_lowpass_filter(trace, freqmax):
            """
            Custom Chebychev type two zerophase lowpass filter useful for
            decimation filtering.

            This filter is stable up to a reduction in frequency with a factor of
            10. If more reduction is desired, simply decimate in steps.

            Partly based on a filter in ObsPy.

            :param trace: The trace to be filtered.
            :param freqmax: The desired lowpass frequency.

            Will be replaced once ObsPy has a proper decimation filter.
            """
            # rp - maximum ripple of passband, rs - attenuation of stopband
            rp, rs, order = 1, 96, 1e99
            ws = freqmax / (
            trace.stats.sampling_rate * 0.5)  # stop band frequency
            wp = ws  # pass band frequency

            while True:
                if order <= 12:
                    break
                wp *= 0.99
                order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=False)

            b, a = signal.cheby2(order, rs, wn, btype="low", analog=False,
                                 output="ba")

            # Apply twice to get rid of the phase distortion.
            trace.data = signal.filtfilt(b, a, trace.data)


        for tr in st:
            samp_rate = tr.stats.sampling_rate
            # Decimation
            while True:
                decimation_factor = int(light_process_parameters["dt"] /
                                        tr.stats.delta)
                # Decimate in steps for larger sample rate reductions.
                if decimation_factor > 8:
                    decimation_factor = 8
                if decimation_factor > 1:
                    new_freq = tr.stats.sampling_rate / float(
                        decimation_factor)
                    if new_freq < light_process_parameters["max_freq"]:
                        print(f"Ich bin hier. new_freq: {new_freq}")
                        print(f"Decimation factor: {decimation_factor}")
                        print(f"Initial sampling: {samp_rate}")
                        print(f"Now sampling: {tr.stats.sampling_rate}")
                        print(f"Trace dt: {tr.stats.delta}")
                        break
                    zerophase_chebychev_lowpass_filter(tr, new_freq)
                    tr.decimate(factor=decimation_factor, no_filter=True)
                else:
                    break
            #new_samp_rate = tr.stats.sampling_rate


        return st

    output_filename = light_process_parameters["temp_file"]
    tag_name = "raw_recording"
    tag_map = {
        "raw_recording": tag_name
    }
    ds.process(light_process_function, output_filename, tag_map=tag_map)

    # replace original raw data with new one
    os.remove(light_process_parameters["event_file_name"])
    os.rename(light_process_parameters["temp_file"],
              light_process_parameters["event_file_name"])
