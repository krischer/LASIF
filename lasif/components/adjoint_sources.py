#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import pyasdf
import os
import numpy as np
from obspy.signal.invsim import cosine_taper

from lasif import LASIFAdjointSourceCalculationError, LASIFNotFoundError
from .component import Component

from ..adjoint_sources.ad_src_tf_phase_misfit import adsrc_tf_phase_misfit
from ..adjoint_sources.ad_src_l2_norm_misfit import adsrc_l2_norm_misfit
from ..adjoint_sources.ad_src_cc_time_shift import adsrc_cc_time_shift
from ..adjoint_sources.ad_src_cc_dd import double_difference_adjoint
from ..adjoint_sources.ad_src_l2_norm_weighted import adsrc_l2_norm_weighted

# Map the adjoint source type names to functions implementing them.
MISFIT_MAPPING = {
    "TimeFrequencyPhaseMisfitFichtner2008": adsrc_tf_phase_misfit,
    "L2Norm": adsrc_l2_norm_misfit,
    "CCTimeShift": adsrc_cc_time_shift,
    "DoubleDifference": double_difference_adjoint,
    "L2NormWeighted": adsrc_l2_norm_weighted
}


class AdjointSourcesComponent(Component):
    """
    Component dealing with the windows and adjoint sources.

    :param folder: The folder where the files are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, folder, communicator, component_name):
        self._folder = folder
        super(AdjointSourcesComponent, self).__init__(
            communicator, component_name)

    def get_filename(self, event, iteration):
        """
        Gets the filename for the adjoint source and windows file.

        :param event: The event.
        :param iteration: The iteration.
        """
        event = self.comm.events.get(event)
        iteration_long_name = self.comm.iterations.get_long_iteration_name(
            iteration)

        folder = os.path.join(self._folder, iteration_long_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        return os.path.join(
            folder, "ADJ_SRC_" + event["event_name"] + ".h5")

    def get_misfit_for_event(self, event, iteration, weight_set_name=None):
        """
        This function returns the total misfit for an event.
        :param event: name of the event
        :param iteration: teration for which to get the misfit
        :return: t
        """
        filename = self.get_filename(event=event, iteration=iteration)

        event_weight = 1.0
        if weight_set_name:
            ws = self.comm.weights.get(weight_set_name)
            event_weight = ws.events[event]["event_weight"]
            station_weights = ws.events[event]["stations"]

        if not os.path.exists(filename):
            raise LASIFNotFoundError(f"Could not find {filename}")

        with pyasdf.ASDFDataSet(filename, mode="r") as ds:
            adj_src_data = ds.auxiliary_data["AdjointSources"]
            stations = ds.auxiliary_data["AdjointSources"].list()

            total_misfit = 0.0
            for station in stations:
                channels = adj_src_data[station].list()
                for channel in channels:
                    if weight_set_name:
                        station_weight = \
                            station_weights[".".join(
                                station.split("_"))]["station_weight"]
                        misfit = \
                            adj_src_data[station][channel].parameters[
                                "misfit"] * station_weight
                    else:
                        misfit = \
                            adj_src_data[station][channel].parameters["misfit"]
                    total_misfit += misfit
        return total_misfit * event_weight

    def write_adjoint_sources(self, event, iteration, adj_sources):
        """
        Write an ASDF file
        """
        filename = self.get_filename(event=event, iteration=iteration)

        print("\nStarting to write adjoint sources to ASDF file ...")

        adj_src_counter = 0
        # print(adj_sources)

        # DANGERZONE: manually disable the MPIfile driver for pyasdf as
        # we are already in MPI but only rank 0 will enter here and that
        # will confuse pyasdf otherwise.
        with pyasdf.ASDFDataSet(filename, mpi=False) as ds:
            for value in adj_sources.values():
                if not value:
                    continue
                for c_id, adj_source in value.items():
                    net, sta, loc, cha = c_id.split(".")
                    ds.add_auxiliary_data(
                        data=adj_source["adj_source"],
                        data_type="AdjointSources",
                        path="%s_%s/Channel_%s_%s" % (net, sta, loc, cha),
                        parameters={"misfit": adj_source["misfit"]})
                    adj_src_counter += 1
        print("Wrote %i adjoint_sources to the ASDF file." % adj_src_counter)

    def calculate_adjoint_source(self, data, synth, starttime, endtime,
                                 min_period, max_period, ad_src_type,
                                 event, station, iteration, envelope,
                                 window_set, plot=False):
        """
        Calculates an adjoint source for a single window.
        Currently not used, might be used later on for adjoint sources
        which are not included in salvus_misfit

        :param event_name: The name of the event.
        :param iteration_name: The name of the iteration.
        :param channel_id: The channel id in the form NET.STA.NET.CHA.
        :param starttime: The starttime of the window.
        :param endtime: The endtime of the window.
        :param ad_src_type: The type of adjoint source. Currently supported
            are ``"TimeFrequencyPhaseMisfitFichtner2008"`` and ``"L2Norm"``.
        """
        # copy because otherwise the passed traces get modified
        data = copy.deepcopy(data)
        synth = copy.deepcopy(synth)

        if ad_src_type not in MISFIT_MAPPING:
            raise LASIFAdjointSourceCalculationError(
                "Adjoint source type '%s' not supported. Supported types: %s"
                % (ad_src_type, ", ".join(MISFIT_MAPPING.keys())))

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
        if ad_src_type != "DoubleDifference":
            for trace in [data, synth]:
                trace.trim(starttime, endtime)
                dt = trace.stats.delta
                len_window = len(trace.data) * dt
                ratio = min_period * 2.0 / len_window
                # Make minimum window length taper 25% of sides
                p = ratio / 2.0
                if p > 1.0:  # For manually picked smaller windows
                    p = 1.0
                window = cosine_taper(len(trace.data), p=p)
                trace.data = trace.data * window
                trace.trim(original_stats.starttime, original_stats.endtime,
                           pad=True, fill_value=0.0)

        # make time axis
        t = np.linspace(0, (original_stats.npts - 1) * original_stats.delta,
                        original_stats.npts)

        #  set data and synthetics, compute actual misfit
        t = np.require(t, dtype="float64", requirements="C")
        data_d = np.require(data.data, dtype="float64", requirements="C")
        synth_d = np.require(synth.data, dtype="float64", requirements="C")
        #  compute misfit and adjoint source
        if ad_src_type == "DoubleDifference":
            window = [starttime, endtime]
            adsrc = double_difference_adjoint(t=t, data=data_d,
                                              synthetic=synth_d,
                                              min_period=min_period,
                                              window=window,
                                              event=event,
                                              station_name=station,
                                              original_stats=original_stats,
                                              iteration=iteration,
                                              comm=self.comm,
                                              window_set=window_set,
                                              plot=plot)
        elif ad_src_type == "L2NormWeighted":
            adsrc = MISFIT_MAPPING[ad_src_type](
                t, data_d, synth_d, min_period=min_period, event=event,
                station=station, envelope=envelope, plot=plot)
        else:
            adsrc = MISFIT_MAPPING[ad_src_type](
                t, data_d, synth_d, min_period=min_period,
                max_period=max_period, plot=plot)

        # Recreate dictionary for clarity.
        ret_val = {
            "adjoint_source": adsrc["adjoint_source"],
            "misfit_value": adsrc["misfit_value"],
            "details": adsrc["details"]
        }
        if plot:
            return

        if not self._validate_return_value(ret_val):
            raise LASIFAdjointSourceCalculationError(
                "Could not calculate adjoint source due to mismatching types.")

        return ret_val

    @staticmethod
    def _validate_return_value(adsrc):
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
