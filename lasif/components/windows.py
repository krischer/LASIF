#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import glob
import os

from lasif import LASIFAdjointSourceCalculationError
from .component import Component

from ..adjoint_sources.ad_src_tf_phase_misfit import adsrc_tf_phase_misfit
from ..adjoint_sources.ad_src_l2_norm_misfit import adsrc_l2_norm_misfit
from ..adjoint_sources.ad_src_cc_time_shift import adsrc_cc_time_shift

import numpy as np

from ..window_manager_sql import WindowGroupManager

# Map the adjoint source type names to functions implementing them.
MISFIT_MAPPING = {
    "TimeFrequencyPhaseMisfitFichtner2008": adsrc_tf_phase_misfit,
    "L2Norm": adsrc_l2_norm_misfit,
    "CCTimeShift": adsrc_cc_time_shift
}


class WindowsAndAdjointSourcesComponent(Component):
    """
    Component dealing with the windows and adjoint sources.

    :param folder: The folder where the files are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, folder, communicator, component_name):
        self._folder = folder
        super(WindowsAndAdjointSourcesComponent, self).__init__(
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
            folder, "ADJ_SRC_AND_WINDOWS_" + event["event_name"] + ".h5")

    def get(self, window_set_name):
        """
        Returns the window manager instance for a window set.

        :param window_set_name: The name of the window set.
        """
        filename = self.get_window_set_filename(window_set_name)
        return WindowGroupManager(filename)

    def list(self):
        """
        Returns a list of window sets currently
        present within the LASIF project.
        """
        files = [os.path.abspath(_i) for _i in glob.iglob(os.path.join(
            self.comm.project.paths["windows"], "*.sqlite"))]
        window_sets = [os.path.splitext(os.path.basename(_i))[0][:]
                       for _i in files]

        return sorted(window_sets)

    def has_window_set(self, window_set_name):
        """
        Checks whether a window set is alreadu defined.
        ReturnsL True or False
        :param window_set_name: name of the window set
        """
        if window_set_name in self.list():
            return True
        return False

    def get_window_set_filename(self, window_set_name):
        """
        Retrieves the filename for a given window set
        :param window_set_name: The name of the window set
        :return: filename of the window set
        """
        filename = os.path.join(self.comm.project.paths['windows'],
                                window_set_name + ".sqlite")
        return filename

    def write_windows_to_sql(self, event_name, window_set_name, windows):
        """
        Writes windows to the sql database
        :param event_name: The name of the event
        :param window_set_name: The name of the window set
        :param windows: The actual windows, structured in a
        dictionary(stations) of dicts(channels) of lists(windowS)
        of tuples (start- and end times)
        """
        window_group_manager = self.get(window_set_name)
        window_group_manager.write_windows(event_name, windows)

    def write_windows(self, event, iteration, windows):
        """
        Write an ASDF file
        """
        import pyasdf

        filename = self.get_filename(event=event, iteration=iteration)

        if os.path.exists(filename):
            raise ValueError("Window and adjoint source file '%s' already "
                             "exists." % filename)

        print("\nStarting to write windows to ASDF file ...")

        win_counter = 0

        # DANGERZONE: manually disable the MPI file driver for pyasdf as
        # we are already in MPI but only rank 0 will enter here and that
        # will confuse pyasdf otherwise.
        with pyasdf.ASDFDataSet(filename, mpi=False) as ds:
            for value in windows.values():
                if not value:
                    continue
                for c_id, windows in value.items():
                    net, sta, loc, cha = c_id.split(".")
                    for _i, window in enumerate(windows):
                        ds.add_auxiliary_data(
                            data=np.empty(0),
                            data_type="Windows",
                            path="%s_%s/Channel_%s_%s/window_%i" % (
                                net, sta, loc, cha, _i),
                            parameters={"starttime": str(window[0]),
                                        "endtime": str(window[1])})
                        win_counter += 1
        print("Wrote %i windows to the ASDF file." % win_counter)

    def write_adjoint_sources(self, event, iteration, adj_sources):
        """
        Write an ASDF file
        """
        import pyasdf

        filename = self.get_filename(event=event, iteration=iteration)

        print("\nStarting to write adjoint sources to ASDF file ...")

        adj_src_counter = 0

        # DANGERZONE: manually disable the MPI file driver for pyasdf as
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

    def read_all_windows(self, event, window_set_name):
        """
        Return a flat dictionary with all windows for a specific event.
        This should always be
        fairly small.
        """
        window_group_manager = self.get(window_set_name)
        return window_group_manager.get_all_windows_for_event(event_name=event)

    def calculate_adjoint_source(self, data, synth, starttime, endtime, taper,
                                 taper_percentage, min_period,
                                 max_period, ad_src_type, plot=False):
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

        #  compute misfit and adjoint source
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
