#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os

from .component import Component

from ..window_manager_sql import WindowGroupManager


class WindowsComponent(Component):
    """
    Component dealing with the windows and adjoint sources.

    :param folder: The folder where the files are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, communicator, component_name):
        super(WindowsComponent, self).__init__(
            communicator, component_name)

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

    def read_all_windows(self, event, window_set_name):
        """
        Return a flat dictionary with all windows for a specific event.
        This should always be
        fairly small.
        """
        window_group_manager = self.get(window_set_name)
        return window_group_manager.get_all_windows_for_event(event_name=event)

    def get_window_statistics(self, window_set_name, events):
        """
        Get a dictionary with window statistics for an iteration per event.
        Depending on the size of your inversion and chosen iteration,
        this might take a while...
        :param window_set_name: The window_set_name.
        :param events: List of event(s)
        """
        statistics = {}

        for _i, event in enumerate(events):
            print("Collecting statistics for event %i of %i ..." % (
                _i + 1, len(events)))

            wm = self.read_all_windows(event=event,
                                       window_set_name=window_set_name)

            # wm is dict with stations/channels/list of start_end tuples
            station_details = \
                self.comm.query.get_all_stations_for_event(event)

            component_window_count = {"E": 0, "N": 0, "Z": 0}
            component_length_sum = {"E": 0, "N": 0, "Z": 0}
            stations_with_windows_count = 0
            stations_without_windows_count = 0
            for station in station_details.keys():
                wins = wm[station] if station in wm.keys() else {}
                has_windows = False
                for channel, windows in wins.items():
                    component = channel[-1].upper()

                    total_length = 0.0
                    for window in windows:
                        total_length += window[1] - window[0]
                    if not total_length > 0.0:
                        continue
                    has_windows = True
                    component_window_count[component] += 1
                    component_length_sum[component] += total_length
                if has_windows:
                    stations_with_windows_count += 1
                else:
                    stations_without_windows_count += 1

            statistics[event] = {
                "total_station_count": len(station_details.keys()),
                "stations_with_windows": stations_with_windows_count,
                "stations_without_windows": stations_without_windows_count,
                "stations_with_vertical_windows": component_window_count["Z"],
                "stations_with_north_windows": component_window_count["N"],
                "stations_with_east_windows": component_window_count["E"],
                "total_window_length": sum(component_length_sum.values()),
                "window_length_vertical_components": component_length_sum["Z"],
                "window_length_north_components": component_length_sum["N"],
                "window_length_east_components": component_length_sum["E"]
            }

        return statistics
