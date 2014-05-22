#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project components class.

It is important to not import necessary things at the method level to make
importing this file as fast as possible. Otherwise using the command line
interface feels sluggish and slow. Import things only the functions they are
needed.

:copyright: Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license: GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import os

from lasif import LASIFError

from .communicator import Communicator
from .component import Component
from .events import EventsComponent
from .stations import StationsComponent
from .waveforms import WaveformsComponent


class Project(Component):
    """
    A class managing LASIF projects.

    It represents the heart of LASIF.
    """
    def __init__(self, project_root_path, init_project=False):
        """
        Upon intialization, set the paths and read the config file.

        :type project_root_path: str
        :param project_root_path: The root path of the project.
        :type init_project: str
        :param init_project: Determines whether or not to initialize a new
            project, e.g. create the necessary folder structure. If a string is
            passed, the project will be given this name. Otherwise a default
            name will be chosen. Defaults to False.
        """
        if init_project:
            self.__init_new_project(init_project)
        if not os.path.exists(self.paths["config_file"]):
            msg = ("Could not find the project's config file. Wrong project "
                   "path or uninitialized project?")
            raise LASIFError(msg)

        # Setup the paths.
        self.__setup_paths(project_root_path)

        # Setup the communicator and register this component.
        super(Project, self).__init__(Communicator(), "project")
        # Setup the different components.
        self.__setup_components()

        # Finally update the folder structure.
        self.__update_folder_structure()

        # self._read_config_file()

    def __setup_components(self):
        """
        Setup the different components of the project. The goal is to
        decouple them as much as possible to keep the structure sane and
        maintainable.

        Communication will happen through the communicator which will also
        keep the references to the single components.
        """
        EventsComponent(folder=self.paths["events"], communicator=self.comm,
                        component_name="events")
        StationsComponent(stationxml_folder=self.paths["station_xml"],
                          seed_folder=self.paths["dataless_seed"],
                          resp_folder=self.paths["resp"],
                          communicator=self.comm, component_name="stations")
        WaveformsComponent(data_folder=self.paths["data"],
                           synthetics_folder=self.paths["synthetics"],
                           communicator=self.comm, component_name="waveforms")

    def __setup_paths(self, root_path):
        """
        Central place to define all paths.
        """
        # Every key containing the string "file" denotes a file, all others
        # should denote directories.
        self.paths = {}
        self.paths["root"] = root_path

        self.paths["events"] = os.path.join(root_path, "EVENTS")
        self.paths["data"] = os.path.join(root_path, "DATA")
        self.paths["cache"] = os.path.join(root_path, "CACHE")
        self.paths["logs"] = os.path.join(root_path, "LOGS")
        self.paths["models"] = os.path.join(root_path, "MODELS")
        self.paths["wavefields"] = os.path.join(root_path, "WAVEFIELDS")
        self.paths["iterations"] = os.path.join(root_path, "ITERATIONS")
        self.paths["synthetics"] = os.path.join(root_path, "SYNTHETICS")
        self.paths["kernels"] = os.path.join(root_path, "KERNELS")
        self.paths["stations"] = os.path.join(root_path, "STATIONS")
        self.paths["output"] = os.path.join(root_path, "OUTPUT")

        self.paths["windows"] = os.path.join(
            root_path, "ADJOINT_SOURCES_AND_WINDOWS", "WINDOWS")
        self.paths["adjoint_sources"] = os.path.join(
            root_path, "ADJOINT_SOURCES_AND_WINDOWS", "ADJOINT_SOURCES")

        # Station file subfolders.
        self.paths["dataless_seed"] = os.path.join(self.paths["stations"],
                                                   "SEED")
        self.paths["station_xml"] = os.path.join(self.paths["stations"],
                                                 "StationXML")
        self.paths["resp"] = os.path.join(self.paths["stations"],
                                          "RESP")

        # Paths for various files.
        self.paths["config_file"] = os.path.join(root_path,
                                                 "config.xml")
        self.paths["config_file_cache"] = \
            os.path.join(self.paths["cache"], "config.xml_cache.pickle")
        self.paths["inv_db_file"] = \
            os.path.join(self.paths["cache"], "inventory_db.sqlite")

    def __update_folder_structure(self):
        """
        Updates the folder structure of the project.
        """
        for name, path in self.paths.iteritems():
            if "file" in name or os.path.exists(path):
                continue
            os.makedirs(path)
        events = self.events.keys()
        folders = [self.paths["data"], self.paths["synthetics"]]
        for folder in folders:
            for event in events:
                event_folder = os.path.join(folder, event)
                if os.path.exists(event_folder):
                    continue
                os.makedirs(event_folder)

    def __init_new_project(self, project_name):
        """
        Initializes a new project. This currently just means that it creates a
        default config file. The folder structure is checked and rebuilt every
        time the project is initialized anyways.
        """
        from lxml import etree
        from lxml.builder import E

        if not project_name:
            project_name = "LASIFProject"

        doc = E.lasif_project(
            E.name(project_name),
            E.description(""),
            E.download_settings(
                E.arclink_username(""),
                E.seconds_before_event(str(300)),
                E.seconds_after_event(str(3600))),
            E.domain(
                E.domain_bounds(
                    E.minimum_longitude(str(-20)),
                    E.maximum_longitude(str(20)),
                    E.minimum_latitude(str(-20)),
                    E.maximum_latitude(str(20)),
                    E.minimum_depth_in_km(str(0.0)),
                    E.maximum_depth_in_km(str(200.0)),
                    E.boundary_width_in_degree(str(3.0))),
                E.domain_rotation(
                    E.rotation_axis_x(str(1.0)),
                    E.rotation_axis_y(str(1.0)),
                    E.rotation_axis_z(str(1.0)),
                    E.rotation_angle_in_degree(str(-45.0)))))

        string_doc = etree.tostring(doc, pretty_print=True,
                                    xml_declaration=True, encoding="UTF-8")

        with open(self.paths["config_file"], "wt") as open_file:
            open_file.write(string_doc)
