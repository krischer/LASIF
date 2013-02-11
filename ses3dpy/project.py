#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project management class.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from obspy import readEvents
import os
from lxml import etree
import matplotlib.pyplot as plt

import visualization


class Project(object):
    def __init__(self, project_root_path):
        """
        """
        self._setup_paths(project_root_path)
        self._read_domain()

    def _read_domain(self):
        """
        Parse the domain definition file.
        """
        domain = etree.parse(self.paths["domain_definition"]).getroot()
        self.domain = {}
        self.domain["bounds"] = {}
        self.domain["name"] = domain.find("name").text
        self.domain["description"] = domain.find("description").text
        if self.domain["description"] is None:
            self.domain["description"] = ""

        bounds = domain.find("domain_bounds")
        self.domain["bounds"]["minimum_latitude"] = \
            float(bounds.find("minimum_latitude").text)
        self.domain["bounds"]["maximum_latitude"] = \
            float(bounds.find("maximum_latitude").text)
        self.domain["bounds"]["minimum_longitude"] = \
            float(bounds.find("minimum_longitude").text)
        self.domain["bounds"]["maximum_longitude"] = \
            float(bounds.find("maximum_longitude").text)
        self.domain["bounds"]["minimum_depth_in_km"] = \
            float(bounds.find("minimum_depth_in_km").text)
        self.domain["bounds"]["maximum_depth_in_km"] = \
            float(bounds.find("maximum_depth_in_km").text)

        rotation = domain.find("domain_rotation")
        self.domain["rotation_axis"] = [
            float(rotation.find("rotation_axis_x").text),
            float(rotation.find("rotation_axis_y").text),
            float(rotation.find("rotation_axis_z").text)]
        self.domain["rotation_angle"] = \
            float(rotation.find("rotation_angle_in_degree").text)

    def plot_domain(self):
        bounds = self.domain["bounds"]
        visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"],
            plot_simulation_domain=True, show_plot=True)

    def read_events(self):
        """
        Parses all events.
        """
        self.events = readEvents(os.path.join(self.paths["events"], "*%sxml" %
            os.path.extsep))

    def plot_events(self, resolution="c"):
        bounds = self.domain["bounds"]
        map = visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"],
            plot_simulation_domain=False, show_plot=False)
        if not hasattr(self, "events") or not self.events:
            self.read_events()
        visualization.plot_events(self.events, map_object=map)
        plt.show()

    def _setup_paths(self, root_path):
        """
        Central place to define all paths
        """
        self.paths = {}
        self.paths["root"] = root_path
        self.paths["domain_definition"] = os.path.join(root_path,
            "simulation_domain.xml")
        self.paths["events"] = os.path.join(root_path, "EVENTS")
