#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

from lasif import LASIFError

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
            domain = self.comm.project.domain
            if domain == "global":
                map = visualization.plot_domain()
            else:
                bounds = domain["bounds"]
                map = visualization.plot_domain(
                    bounds["minimum_latitude"], bounds["maximum_latitude"],
                    bounds["minimum_longitude"], bounds["maximum_longitude"],
                    bounds["boundary_width_in_degree"],
                    rotation_axis=domain["rotation_axis"],
                    rotation_angle_in_degree=domain["rotation_angle"],
                    plot_simulation_domain=False, zoom=True)
            visualization.plot_events(events, map_object=map, project=self)
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

        from lasif import visualization

        # Plot the domain.
        domain = self.comm.project.domain
        if domain == "global":
            map_object = visualization.plot_domain()
        else:
            bounds = domain["bounds"]
            map_object = visualization.plot_domain(
                bounds["minimum_latitude"], bounds["maximum_latitude"],
                bounds["minimum_longitude"], bounds["maximum_longitude"],
                bounds["boundary_width_in_degree"],
                rotation_axis=domain["rotation_axis"],
                rotation_angle_in_degree=domain["rotation_angle"],
                plot_simulation_domain=False, zoom=True)

        # Get the event and extract information from it.
        event_info = self.comm.events.get(event_name)

        # Get a dictionary containing all stations that have data for the
        # current event.
        stations = self.comm.query.get_all_stations_for_event(event_name)

        # Plot the stations. This will also plot raypaths.
        visualization.plot_stations_for_event(
            map_object=map_object, station_dict=stations,
            event_info=event_info, project=self)

        # Plot the beachball for one event.
        visualization.plot_events([event_info], map_object=map_object)

    def plot_domain(self):
        """
        Plots the simulation domain and the actual physical domain.

        Wrapper around one of the visualization routines.
        """
        from lasif import visualization

        domain = self.comm.project.domain
        if domain == "global":
            visualization.plot_domain()
        else:
            bounds = domain["bounds"]
            visualization.plot_domain(
                bounds["minimum_latitude"],
                bounds["maximum_latitude"], bounds["minimum_longitude"],
                bounds["maximum_longitude"],
                bounds["boundary_width_in_degree"],
                rotation_axis=domain["rotation_axis"],
                rotation_angle_in_degree=domain["rotation_angle"],
                plot_simulation_domain=True, zoom=True)

    def plot_raydensity(self, save_plot=True):
        """
        Plots the raydensity.
        """
        from lasif import visualization
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 21))

        domain = self.comm.project.domain
        bounds = domain["bounds"]
        map_object = visualization.plot_domain(
            bounds["minimum_latitude"], bounds["maximum_latitude"],
            bounds["minimum_longitude"], bounds["maximum_longitude"],
            bounds["boundary_width_in_degree"],
            rotation_axis=domain["rotation_axis"],
            rotation_angle_in_degree=domain["rotation_angle"],
            plot_simulation_domain=False, zoom=True,
            resolution="l")

        event_stations = []
        for event_name, event_info in \
                self.comm.events.get_all_events().iteritems():
            try:
                stations = \
                    self.comm.query.get_all_stations_for_event(event_name)
            except LASIFError:
                stations = {}
            event_stations.append((event_info, stations))

        visualization.plot_raydensity(
            map_object, event_stations, bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], domain["rotation_axis"],
            domain["rotation_angle"])

        visualization.plot_events(self.comm.events.get_all_events().values(),
                                  map_object=map_object)

        plt.tight_layout()

        if save_plot:
            outfile = os.path.join(
                self.comm.project.get_output_folder("raydensity_plot"),
                "raydensity.png")
            plt.savefig(outfile, dpi=200, transparent=True)
            print "Saved picture at %s" % outfile

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


