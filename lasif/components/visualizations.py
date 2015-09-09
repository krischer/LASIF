#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import itertools
import math
import numpy as np
import os

from lasif import LASIFError, LASIFNotFoundError

from .component import Component


class VisualizationsComponent(Component):
    """
    Component offering project visualization. Has to be initialized fairly
    late at is requires a lot of data to be present.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def plot_events(self, plot_type="map"):
        """
        Plots the domain and beachballs for all events on the map.

        :param plot_type: Determines the type of plot created.
            * ``map`` (default) - a map view of the events
            * ``depth`` - a depth distribution histogram
            * ``time`` - a time distribution histogram
        """
        from lasif import visualization

        events = self.comm.events.get_all_events().values()

        if plot_type == "map":
            m = self.comm.project.domain.plot()
            visualization.plot_events(events, map_object=m)
        elif plot_type == "depth":
            visualization.plot_event_histogram(events, "depth")
        elif plot_type == "time":
            visualization.plot_event_histogram(events, "time")
        else:
            msg = "Unknown plot_type"
            raise LASIFError(msg)

    def plot_event(self, event_name):
        """
        Plots information about one event on the map.
        """
        if not self.comm.events.has_event(event_name):
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)

        map_object = self.comm.project.domain.plot()

        from lasif import visualization

        # Get the event and extract information from it.
        event_info = self.comm.events.get(event_name)

        # Get a dictionary containing all stations that have data for the
        # current event.
        try:
            stations = self.comm.query.get_all_stations_for_event(event_name)
        except LASIFNotFoundError:
            pass
        else:
            # Plot the stations if it has some. This will also plot raypaths.
            visualization.plot_stations_for_event(
                map_object=map_object, station_dict=stations,
                event_info=event_info)

        # Plot the beachball for one event.
        visualization.plot_events(events=[event_info], map_object=map_object)

    def plot_domain(self, plot_simulation_domain=True):
        """
        Plots the simulation domain and the actual physical domain.
        """
        self.comm.project.domain.plot(
            plot_simulation_domain=plot_simulation_domain)

    def plot_raydensity(self, save_plot=True, plot_stations=False):
        """
        Plots the raydensity.
        """
        from lasif import visualization
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 21))

        map_object = self.comm.project.domain.plot()

        event_stations = []
        for event_name, event_info in \
                self.comm.events.get_all_events().iteritems():
            try:
                stations = \
                    self.comm.query.get_all_stations_for_event(event_name)
            except LASIFError:
                stations = {}
            event_stations.append((event_info, stations))

        visualization.plot_raydensity(map_object=map_object,
                                      station_events=event_stations,
                                      domain=self.comm.project.domain)

        visualization.plot_events(self.comm.events.get_all_events().values(),
                                  map_object=map_object)

        if plot_stations:
            stations = itertools.chain.from_iterable((
                _i[1].values() for _i in event_stations if _i[1]))
            # Remove duplicates
            stations = [(_i["latitude"], _i["longitude"]) for _i in stations]
            stations = set(stations)
            x, y = map_object([_i[1] for _i in stations],
                              [_i[0] for _i in stations])
            map_object.scatter(x, y, s=14 ** 2, color="#333333",
                               edgecolor="#111111", alpha=0.6, zorder=200,
                               marker="v")

        plt.tight_layout()

        if save_plot:
            outfile = os.path.join(
                self.comm.project.get_output_folder(
                    type="raydensity_plots", tag="raydensity"),
                "raydensity.png")
            plt.savefig(outfile, dpi=200, transparent=True)
            print "Saved picture at %s" % outfile

    def plot_windows(self, event, iteration, distance_bins=100,
                     ax=None, show=True):
        """
        Plot all selected windows on a epicentral distance vs duration plot
        with the color encoding the selected channels. This gives a quick
        overview of how well selected the windows for a certain event and
        iteration are.

        :param event: The event.
        :param iteration: The iteration.
        :param distance_bins: The number of bins on the epicentral
            distance axis.
        :param ax: If given, it will be plotted to this ax.
        :param show: If true, ``plt.show()`` will be called before returning.
        :return: The potentially created axes object.
        """
        from obspy.geodetics.base import locations2degrees

        event = self.comm.events.get(event)
        iteration = self.comm.iterations.get(iteration)
        pparam = iteration.get_process_params()
        window_manager = self.comm.windows.get(event, iteration)

        starttime = event["origin_time"]
        duration = (pparam["npts"] - 1) * pparam["dt"]

        # First step is to calculate all epicentral distances.
        stations = copy.deepcopy(self.comm.query.get_all_stations_for_event(
            event["event_name"]))
        for s in stations.values():
            s["epicentral_distance"] = locations2degrees(
                event["latitude"], event["longitude"], s["latitude"],
                s["longitude"])

        min_epicentral_distance = math.floor(min(
            _i["epicentral_distance"] for _i in stations.values()))
        max_epicentral_distance = math.ceil(max(
            _i["epicentral_distance"] for _i in stations.values()))
        epicentral_range = max_epicentral_distance - min_epicentral_distance

        if epicentral_range == 0:
            raise ValueError

        # Create the image that will represent the pictures in an epicentral
        # distance plot. By default everything is black.
        #
        # First dimension: Epicentral distance.
        # Second dimension: Time.
        # Third dimension: RGB tuple.
        len_time = 1000
        len_dist = distance_bins
        image = np.zeros((len_dist, len_time, 3), dtype=np.uint8)

        # Helper functions calculating the indices.
        def _time_index(value):
            frac = np.clip((value - starttime) / duration, 0, 1)
            return int(round(frac * len_time))

        def _space_index(value):
            frac = np.clip(
                (value - min_epicentral_distance) / epicentral_range, 0, 1)
            return int(round(frac * len_dist))

        def _color_index(channel):
            _map = {
                "Z": 2,
                "N": 1,
                "E": 0
            }
            channel = channel[-1].upper()
            if channel not in _map:
                raise ValueError
            return _map[channel]

        for channel in window_manager.list():
            station = ".".join(channel.split(".")[:2])
            for win in window_manager.get(channel):
                image[
                    _space_index(stations[station]["epicentral_distance"]),
                    _time_index(win.starttime):_time_index(win.endtime),
                    _color_index(channel)] = 255

        # From http://colorbrewer2.org/
        color_map = {
            (255, 0, 0): (228, 26, 28),  # red
            (0, 255, 0): (77, 175, 74),  # green
            (0, 0, 255): (55, 126, 184),  # blue
            (255, 0, 255): (152, 78, 163),  # purple
            (0, 255, 255): (255, 127, 0),  # orange
            (255, 255, 0): (255, 255, 51),  # yellow
            (255, 255, 255): (250, 250, 250),  # white
            (0, 0, 0): (50, 50, 50)  # More pleasent gray background
        }

        # Replace colors...fairly complex. Not sure if there is another way...
        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        for color, replacement in color_map.items():
            image[:, :, :][(red == color[0]) & (green == color[1]) &
                           (blue == color[2])] = replacement

        def _one(i):
            return [_i / 255.0 for _i in i]

        import matplotlib.pylab as plt
        plt.style.use("ggplot")

        artists = [
            plt.Rectangle((0, 1), 1, 1, color=_one(color_map[(0, 0, 255)])),
            plt.Rectangle((0, 1), 1, 1, color=_one(color_map[(0, 255, 0)])),
            plt.Rectangle((0, 1), 1, 1, color=_one(color_map[(255, 0, 0)])),
            plt.Rectangle((0, 1), 1, 1, color=_one(color_map[(0, 255, 255)])),
            plt.Rectangle((0, 1), 1, 1, color=_one(color_map[(255, 0, 255)])),
            plt.Rectangle((0, 1), 1, 1, color=_one(color_map[(255, 255, 0)])),
            plt.Rectangle((0, 1), 1, 1,
                          color=_one(color_map[(255, 255, 255)]))
        ]
        labels = [
            "Z",
            "N",
            "E",
            "Z + N",
            "Z + E",
            "N + E",
            "Z + N + E"
        ]

        if ax is None:
            plt.figure(figsize=(16, 9))
            ax = plt.gca()

        ax.imshow(image, aspect="auto", interpolation="nearest", vmin=0,
                  vmax=255, origin="lower")
        ax.grid()
        ax.set_title("Selected windows for iteration %s and event %s" % (
                     iteration.name, event["event_name"]))

        ax.legend(artists, labels, loc="lower right",
                  title="Selected Components")

        # Set the x-ticks.
        xticks = []
        for time in ax.get_xticks():
            # They are offset by -0.5.
            time += 0.5
            # Convert to actual time
            frac = time / float(len_time)
            time = frac * duration
            xticks.append("%.1f" % time)
        ax.set_xticklabels(xticks)
        ax.set_xlabel("Time since event in seconds")

        yticks = []
        for dist in ax.get_yticks():
            # They are offset by -0.5.
            dist += 0.5
            # Convert to actual epicentral distance.
            frac = dist / float(len_dist)
            dist = min_epicentral_distance + (frac * epicentral_range)
            yticks.append("%.1f" % dist)
        ax.set_yticklabels(yticks)
        ax.set_ylabel("Epicentral distance in degree [Binned in %i distances]"
                      % distance_bins)

        if show:
            plt.tight_layout()
            plt.show()
            plt.close()

        return ax

    def plot_data_and_synthetics(self, event, iteration, channel_id, ax=None,
                                 show=True):
        """
        Plots the data and corresponding synthetics for a given event,
        iteration, and channel.

        :param event: The event.
        :param iteration: The iteration.
        :param channel_id: The channel id.
        :param ax: If given, it will be plotted to this ax.
        :param show: If true, ``plt.show()`` will be called before returning.
        :return: The potentially created axes object.
        """
        import matplotlib.pylab as plt

        data = self.comm.query.get_matching_waveforms(event, iteration,
                                                      channel_id)
        if ax is None:
            plt.figure(figsize=(15, 3))
            ax = plt.gca()

        iteration = self.comm.iterations.get(iteration)

        ax.plot(data.data[0].times(), data.data[0].data, color="black",
                label="observed")
        ax.plot(data.synthetics[0].times(), data.synthetics[0].data,
                color="red",
                label="synthetic, iteration %s" % str(iteration.name))
        ax.legend()

        ax.set_xlabel("Seconds since event")
        ax.set_ylabel("m/s")
        ax.set_title(channel_id)
        ax.grid()

        if iteration.scale_data_to_synthetics:
            ax.text(0.995, 0.005, "data scaled to synthetics",
                    horizontalalignment="right", verticalalignment="bottom",
                    transform=ax.transAxes, color="0.2")

        if show:
            plt.tight_layout()
            plt.show()
            plt.close()

        return ax
