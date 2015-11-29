#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import json
import os
import shutil

from lasif import LASIFNotFoundError
from .component import Component
from ..window_manager import WindowGroupManager

import numpy as np


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
        iteration = self.comm.iterations.get(iteration)

        folder = os.path.join(self._folder, iteration.long_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        return os.path.join(
            folder, "ADJ_SRC_AND_WINDOWS_" + event["event_name"] + ".h5")

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
            for value in windows.itervalues():
                if not value:
                    continue
                for c_id, windows in value.iteritems():
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

    # def list(self):
    #     """
    #     Lists all events with windows.
    #     """
    #     return sorted([_i for _i in os.listdir(self._folder) if
    #                    os.path.isdir(os.path.join(self._folder, _i))])

    # def list_for_event(self, event_name):
    #     """
    #     List of all iterations with windows for an event.
    #
    #     :param event_name: The name of the event.
    #     """
    #     event_folder = os.path.join(self._folder, event_name)
    #     if not os.path.exists(event_folder):
    #         msg = "No windows for event '%s'." % event_name
    #         raise LASIFNotFoundError(msg)
    #
    #     return sorted([_i.lstrip("ITERATION_")
    #                    for _i in os.listdir(event_folder) if
    #                    os.path.isdir(os.path.join(event_folder, _i)) and
    #                    _i.startswith("ITERATION_")])

    # def get(self, event, iteration):
    #     """
    #     Returns the window manager instance for a given event and iteration.
    #
    #     :param event_name: The name of the event.
    #     :param iteration_name: The name of the iteration.
    #     """
    #     event = self.comm.events.get(event)
    #     iteration = self.comm.iterations.get(iteration)
    #     event_name = event["event_name"]
    #     iteration_name = iteration.name
    #
    #     if not self.comm.events.has_event(event_name):
    #         msg = "Event '%s' not known." % event_name
    #         raise LASIFNotFoundError(msg)
    #
    #     if not self.comm.iterations.has_iteration(iteration_name):
    #         msg = "Iteration '%s' not known." % iteration_name
    #         raise LASIFNotFoundError(msg)
    #
    #     iteration = self.comm.iterations.get(iteration_name)
    #     if event_name not in iteration.events:
    #         msg = "Event '%s' not part of iteration '%s'." % (event_name,
    #                                                           iteration_name)
    #         raise ValueError(msg)
    #
    #     folder = os.path.join(self._folder, event_name,
    #                           self.comm.iterations.get_long_iteration_name(
    #                               iteration_name))
    #     return WindowGroupManager(folder, iteration_name, event_name,
    #                               comm=self.comm)
    #
    # def _clear_window_statistics_cache(self):
    #     """
    #     Clears the window statistics cache.
    #     """
    #     if os.path.exists(self._statistics_cache_folder):
    #         shutil.rmtree(self._statistics_cache_folder)
    #
    #     if not os.path.exists(self._statistics_cache_folder):
    #         os.makedirs(self._statistics_cache_folder)
    #
    # def get_window_statistics(self, iteration, cache=True):
    #     """
    #     Get a dictionary with window statistics for an iteration per event.
    #
    #     Depending on the size of your inversion and chosen iteration,
    #     this might take a while...
    #
    #     :param iteration: The iteration for which to calculate everything.
    #     :param cache: Use cache (if available). Otherwise cached value will
    #         be deleted.
    #     """
    #     from obspy.geodetics.base import locations2degrees
    #
    #     it = self.comm.iterations.get(iteration)
    #
    #     cache_file = os.path.join(self._statistics_cache_folder,
    #                               "window_statistics_iteration_%s.json" %
    #                               it.name)
    #
    #     if os.path.exists(cache_file):
    #         if cache is True:
    #             try:
    #                 with open(cache_file) as fh:
    #                     data = json.load(fh)
    #             except Exception as e:
    #                 print("Loading cache failed due to: %s" % str(e))
    #             print("Loading statistics from cache.")
    #             return data
    #         else:
    #             print("Removing existing cached file ...")
    #             os.remove(cache_file)
    #
    #     statistics = {}
    #
    #     for _i, event in enumerate(list(sorted(it.events.keys()))):
    #         print("Collecting statistics for event %i of %i ..." % (
    #             _i + 1, len(it.events)))
    #
    #         wm = self.get(event=event, iteration=iteration)
    #
    #         event_obj = self.comm.events.get(event)
    #         station_details = copy.deepcopy(
    #             self.comm.query.get_all_stations_for_event(event))
    #
    #         component_window_count = {"E": 0, "N": 0, "Z": 0}
    #         component_length_sum = {"E": 0, "N": 0, "Z": 0}
    #         stations_with_windows_count = 0
    #         stations_without_windows_count = 0
    #
    #         stations = {}
    #
    #         for station in it.events[event]["stations"].keys():
    #             s = station_details[station]
    #             stations[station] = s
    #
    #             s["epicentral_distance"] = locations2degrees(
    #                 event_obj["latitude"], event_obj["longitude"],
    #                 s["latitude"], s["longitude"])
    #
    #             s["windows"] = {"Z": [], "E": [], "N": []}
    #
    #             wins = wm.get_windows_for_station(station)
    #             has_windows = False
    #             for coll in wins:
    #                 component = coll.channel_id[-1].upper()
    #                 total_length = sum([_i.length for _i in coll.windows])
    #                 if not total_length:
    #                     continue
    #                 for win in coll.windows:
    #                     s["windows"][component].append(win.length)
    #                 has_windows = True
    #                 component_window_count[component] += 1
    #                 component_length_sum[component] += total_length
    #             if has_windows:
    #                 stations_with_windows_count += 1
    #             else:
    #                 stations_without_windows_count += 1
    #
    #         statistics[event] = {
    #             "total_station_count": len(it.events[event]["stations"]),
    #             "stations_with_windows": stations_with_windows_count,
    #             "stations_without_windows": stations_without_windows_count,
    #             "stations_with_vertical_windows": component_window_count["Z"],
    #             "stations_with_north_windows": component_window_count["N"],
    #             "stations_with_east_windows": component_window_count["E"],
    #             "total_window_length": sum(component_length_sum.values()),
    #             "window_length_vertical_components": component_length_sum["Z"],
    #             "window_length_north_components": component_length_sum["N"],
    #             "window_length_east_components": component_length_sum["E"],
    #             "stations": stations
    #         }
    #
    #     # Store in cache.
    #     with open(cache_file, "w") as fh:
    #         json.dump(statistics, fh)
    #
    #     return statistics
