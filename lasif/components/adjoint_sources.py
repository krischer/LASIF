#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import joblib
import numpy as np
import os

from lasif import LASIFNotFoundError, LASIFAdjointSourceCalculationError
from .component import Component
from ..adjoint_sources.ad_src_tf_phase_misfit import adsrc_tf_phase_misfit
from ..adjoint_sources.ad_src_l2_norm_misfit import adsrc_l2_norm_misfit
from ..adjoint_sources.ad_src_cc_time_shift import adsrc_cc_time_shift


# Map the adjoint source type names to functions implementing them.
MISFIT_MAPPING = {
    "TimeFrequencyPhaseMisfitFichtner2008": adsrc_tf_phase_misfit,
    "L2Norm": adsrc_l2_norm_misfit,
    "CCTimeShift": adsrc_cc_time_shift
}


class AdjointSourcesComponent(Component):
    """
    Component dealing with the adjoint sources.

    :param ad_src_folder: The folder where the adjoint sources are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, ad_src_folder, communicator, component_name):
        self._folder = ad_src_folder
        super(AdjointSourcesComponent, self).__init__(
            communicator, component_name)

    def calculate_adjoint_source(self, event_name, iteration_name,
                                 channel_id, starttime, endtime, taper,
                                 taper_percentage, ad_src_type, plot=False):
        """
        Calculates an adjoint source for a single window.

        :param event_name: The name of the event.
        :param iteration_name: The name of the iteration.
        :param channel_id: The channel id in the form NET.STA.NET.CHA.
        :param starttime: The starttime of the window.
        :param endtime: The endtime of the window.
        :param taper: How to taper the window.
        :param taper_percentage: The taper percentage at one end as a
            decimal number ranging from 0.0 to 0.5 for a full width taper.
        :param ad_src_type: The type of adjoint source. Currently supported
            are ``"TimeFrequencyPhaseMisfitFichtner2008"`` and ``"L2Norm"``.
        """
        iteration = self.comm.iterations.get(iteration_name)
        iteration_name = iteration.long_name
        event = self.comm.events.get(event_name)
        event_name = event["event_name"]

        folder = os.path.join(self._folder, event_name, iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, "%s_%s_%s_%s_%.2f_%s" % (
            channel_id, str(starttime), str(endtime), str(taper),
            taper_percentage, ad_src_type))

        if not plot and os.path.exists(filename):
            adsrc = joblib.load(filename)
            if not self._validate_return_value(adsrc):
                os.remove(filename)
            else:
                return adsrc

        if ad_src_type not in MISFIT_MAPPING:
            raise LASIFAdjointSourceCalculationError(
                "Adjoint source type '%s' not supported. Supported types: %s"
                % (ad_src_type, ", ".join(MISFIT_MAPPING.keys())))

        waveforms = self.comm.query.get_matching_waveforms(
            event=event_name, iteration=iteration_name,
            station_or_channel_id=channel_id)
        data = waveforms.data
        synth = waveforms.synthetics

        if len(data) != 1:
            raise LASIFNotFoundError(
                "Data not found for event '%s', iteration '%s', and channel "
                "'%s'." % (event_name, iteration_name, channel_id))
        if len(synth) != 1:
            raise LASIFNotFoundError(
                "Synthetics not found for event '%s', iteration '%s', "
                "and channel '%s'." % (event_name, iteration_name, channel_id))
        data = data[0]
        synth = synth[0]

        # Make sure they are equal enough.
        if abs(data.stats.starttime - synth.stats.starttime) > 0.1:
            raise LASIFAdjointSourceCalculationError(
                "Starttime not similar enough")
        if data.stats.npts != synth.stats.npts:
            raise LASIFAdjointSourceCalculationError(
                "Differing number of samples")
        if abs(data.stats.delta - synth.stats.delta) / data.stats.delta > \
                0.01:
            raise LASIFAdjointSourceCalculationError(
                "Sampling rate not similar enough.")

        original_stats = copy.deepcopy(data.stats)

        for trace in [data, synth]:
            trace.trim(starttime, endtime)
            trace.taper(type=taper.lower(), max_percentage=taper_percentage)
            trace.trim(original_stats.starttime, original_stats.endtime,
                       pad=True, fill_value=0.0)

        #  make time axis
        t = np.linspace(0, (original_stats.npts - 1) * original_stats.delta,
                        original_stats.npts)

        #  set data and synthetics, compute actual misfit
        t = np.require(t, dtype="float64", requirements="C")
        data_d = np.require(data.data, dtype="float64", requirements="C")
        synth_d = np.require(synth.data, dtype="float64", requirements="C")

        process_parameters = iteration.get_process_params()

        #  compute misfit and adjoint source
        adsrc = MISFIT_MAPPING[ad_src_type](
            t, data_d, synth_d,
            1.0 / process_parameters["lowpass"],
            1.0 / process_parameters["highpass"], plot=plot,
            max_criterion=self.comm.project.config["misc_settings"][
                "time_frequency_adjoint_source_criterion"]
        )
        if plot:
            return

        # Recreate dictionary for clarity.
        ret_val = {
            "adjoint_source": adsrc["adjoint_source"],
            "misfit_value": adsrc["misfit_value"],
            "details": adsrc["details"]
        }

        # If the adjoint source has not been calculated, the misfit might
        # still have. Don't store the adjoint source in that case.
        if ret_val["adjoint_source"] is None and \
                isinstance(ret_val["misfit_value"], float):
            return ret_val

        if not self._validate_return_value(ret_val):
            raise LASIFAdjointSourceCalculationError(
                "Could not calculate adjoint source due to mismatching types.")

        joblib.dump(ret_val, filename)
        return ret_val

    def _validate_return_value(self, adsrc):
        if not isinstance(adsrc, dict):
            return False
        elif sorted(adsrc.keys()) != ["adjoint_source", "details",
                                      "misfit_value"]:
            return False
        elif not isinstance(adsrc["adjoint_source"], np.ndarray):
            return False
        elif not isinstance(adsrc["misfit_value"], float):
            return False
        elif not isinstance(adsrc["details"], dict):
            return False
        return True
