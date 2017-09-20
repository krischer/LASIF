#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project specific function processing observed data.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013-2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
from lasif import LASIFError
from scipy import signal
from pyasdf import ASDFDataSet


def preprocessing_function_asdf(processing_info):

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
        ws = freqmax / (trace.stats.sampling_rate * 0.5)  # stop band frequency
        wp = ws  # pass band frequency

        while True:
            if order <= 12:
                break
            wp *= 0.99
            order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)

        b, a = signal.cheby2(order, rs, wn, btype="low", analog=0, output="ba")

        # Apply twice to get rid of the phase distortion.
        trace.data = signal.filtfilt(b, a, trace.data)

    # =========================================================================
    # Read ASDF file
    # =========================================================================

    ds = ASDFDataSet(processing_info["asdf_input_filename"], compression=None)
    event = ds.events[0]

    # Get processing_info
    npts = processing_info["npts"]
    sampling_rate = 1.0 / processing_info["dt"]
    min_period = processing_info["highpass_period"]
    max_period = processing_info["lowpass_period"]

    origin = event.preferred_origin() or event.origins[0]
    starttime = origin.time + processing_info["salvus_start_time"]
    endtime = starttime + processing_info["dt"] * (npts - 1)
    duration = endtime - starttime

    f2 = 0.9 / max_period
    f3 = 1.1 / min_period
    # Recommendations from the SAC manual.
    f1 = 0.5 * f2
    f4 = 2.0 * f3
    pre_filt = (f1, f2, f3, f4)

    def process_function(st, inv):
        for tr in st:
            # Trim to reduce processing costs
            tr.trim(starttime - 0.2 * duration, endtime + 0.2 * duration)

            # Decimation
            while True:
                decimation_factor = int(processing_info["dt"] /
                                        tr.stats.delta)
                # Decimate in steps for large sample rate reductions.
                if decimation_factor > 8:
                    decimation_factor = 8
                if decimation_factor > 1:
                    new_nyquist = tr.stats.sampling_rate / 2.0 / float(
                        decimation_factor)
                    zerophase_chebychev_lowpass_filter(tr, new_nyquist)
                    tr.decimate(factor=decimation_factor, no_filter=True)
                else:
                    break

        # Detrend and taper
        st.detrend("linear")
        st.detrend("demean")
        st.taper(max_percentage=0.05, type="hann")

        # Instrument correction
        try:
            st.attach_response(inv)
            st.remove_response(output="DISP", pre_filt=pre_filt,
                               zero_mean=False, taper=False)
        except Exception as e:
            net = inv.get_contents()['channels'][0].split('.', 2)[0]
            sta = inv.get_contents()['channels'][0].split('.', 2)[1]

            msg = ("Station: %s.%s could not be corrected with the help of "
                   "asdf file: '%s'. Due to: '%s'  Will be skipped.") \
                % (net, sta,
                   processing_info["asdf_input_filename"], e.__repr__()),
            raise LASIFError(msg)

        # Bandpass filtering
        st.detrend("linear")
        st.detrend("demean")
        st.taper(0.05, type="cosine")
        st.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=True)

        st.detrend("linear")
        st.detrend("demean")
        st.taper(0.05, type="cosine")
        st.filter("bandpass", freqmin=1.0 / max_period,
                  freqmax=1.0 / min_period, corners=3, zerophase=True)

        # Sinc interpolation
        for tr in st:
            tr.data = np.require(tr.data, requirements="C")

        st.interpolate(sampling_rate=sampling_rate, method="lanczos",
                       starttime=starttime, window="blackman", a=12, npts=npts)

        # Convert to single precision to save space.
        for tr in st:
            tr.data = np.require(tr.data, dtype="float32", requirements="C")

        return st

    tag_name = processing_info["preprocessing_tag"]

    tag_map = {
        "raw_recording": tag_name
    }

    output_filename = processing_info["asdf_output_filename"]
    ds.process(process_function, output_filename, tag_map=tag_map)

    del ds
