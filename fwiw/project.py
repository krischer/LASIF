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
import glob
from obspy import readEvents
import os
from lxml import etree
import matplotlib.pyplot as plt

import visualization


class FWIWException(Exception):
    pass


class Project(object):
    def __init__(self, project_root_path):
        """
        Upon intialization, set the paths and read the config file.
        """
        self._setup_paths(project_root_path)
        self._read_config_file()

    def __str__(self):
        """
        """
        ret_str = "FWIW project \"%s\"\n" % self.config["name"]
        ret_str += "\tDescription: %s\n" % self.config["description"]
        ret_str += "\tProject root: %s\n" % self.paths["root"]
        return ret_str

    def _read_config_file(self):
        """
        Parse the config file.
        """
        root = etree.parse(self.paths["config_file"]).getroot()

        self.config = {}
        self.config["name"] = root.find("name").text
        self.config["description"] = root.find("description").text
        # The description field is the only field allowed to be empty.
        if self.config["description"] is None:
            self.config["description"] = ""

        self.config["download_settings"] = {}
        dl_settings = root.find("download_settings")
        self.config["download_settings"]["arclink_username"] = \
            dl_settings.find("arclink_username").text
        self.config["download_settings"]["seconds_before_event"] = \
            float(dl_settings.find("seconds_before_event").text)
        self.config["download_settings"]["seconds_after_event"] = \
            float(dl_settings.find("seconds_after_event").text)

        # Read the domain.
        domain = root.find("domain")
        self.domain = {}
        self.domain["bounds"] = {}

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
        self.domain["bounds"]["boundary_width_in_degree"] = \
            float(bounds.find("boundary_width_in_degree").text)

        rotation = domain.find("domain_rotation")
        self.domain["rotation_axis"] = [
            float(rotation.find("rotation_axis_x").text),
            float(rotation.find("rotation_axis_y").text),
            float(rotation.find("rotation_axis_z").text)]
        self.domain["rotation_angle"] = \
            float(rotation.find("rotation_angle_in_degree").text)

    def get_event_dict(self):
        """
        Returns a dictonary with all events in the project, the keys are the
        event names and the values the full paths to each event.
        """
        events = {}
        for event in glob.iglob(os.path.join(self.paths["events"],
                "*%sxml" % os.extsep)):
            event = os.path.abspath(event)
            event_name = os.path.splitext(os.path.basename(event))[0]
            events[event_name] = event
        return events

    def plot_domain(self):
        bounds = self.domain["bounds"]
        visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], bounds["boundary_width_in_degree"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"],
            plot_simulation_domain=True, show_plot=True)

    def read_events(self):
        """
        Parses all events to a catalog object and stores it in self.events.
        """
        self.events = readEvents(os.path.join(self.paths["events"], "*%sxml" %
            os.path.extsep))

    def plot_events(self, resolution="c"):
        bounds = self.domain["bounds"]
        map = visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], bounds["boundary_width_in_degree"],
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
        self.paths["config_file"] = os.path.join(root_path,
            "config.xml")
        self.paths["events"] = os.path.join(root_path, "EVENTS")
        self.paths["data"] = os.path.join(root_path, "DATA")
        self.paths["logs"] = os.path.join(root_path, "LOGS")
        self.paths["synthetics"] = os.path.join(root_path, "SYNTHETICS")
        self.paths["stations"] = os.path.join(root_path, "STATIONS")
