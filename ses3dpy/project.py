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
import os
from lxml import etree

import rotations


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

        self._get_domain_extensions()

    def _get_domain_extensions(self):
        """
        Calculated the maximum extension of the domain in latitude and
        longitude coordinates.
        """
        bounds = self.domain["bounds"]
        bounds = rotations.get_max_extention_of_domain(
            bounds["minimum_latitude"],
            bounds["maximum_latitude"],
            bounds["minimum_longitude"],
            bounds["maximum_longitude"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"])
        extends = {}
        self.domain["calculated_extension"] = extends
        extends["minimum_latitude"] = bounds["minimum_latitude"]
        extends["maximum_latitude"] = bounds["maximum_latitude"]
        extends["minimum_longitude"] = bounds["minimum_longitude"]
        extends["maximum_longitude"] = bounds["maximum_longitude"]
        extends["center_latitude"] = bounds["minimum_latitude"] + \
                (bounds["maximum_latitude"] - bounds["minimum_latitude"]) / 2.0
        extends["center_longitude"] = bounds["minimum_longitude"] + \
                (bounds["maximum_longitude"] - bounds["minimum_longitude"]) \
                / 2.0

    def _setup_paths(self, root_path):
        """
        Central place to define all paths
        """
        self.paths = {}
        self.paths["root"] = root_path
        self.paths["domain_definition"] = os.path.join(root_path,
            "simulation_domain.xml")
