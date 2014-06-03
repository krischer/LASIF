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
