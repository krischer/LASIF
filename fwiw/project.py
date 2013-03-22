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
from lxml import etree
from lxml.builder import E
import obspy
from obspy.xseed import Parser
import os
import matplotlib.pyplot as plt

from fwiw import visualization


class FWIWException(Exception):
    pass


class Project(object):
    """
    A class representing and managing a single FWIW project.
    """
    def __init__(self, project_root_path, init_project=False):
        """
        Upon intialization, set the paths and read the config file.

        :type project_root_path: String
        :param project_root_path: The root path of the project.
        :type init_project: False or String
        :param init_project: Determines whether or not to initialize a new
            project, e.g. create the necessary folder structure. If a string is
            passed, the project will be given this name. Otherwise a default
            name will be chosen.
        """
        self._setup_paths(project_root_path)
        if init_project:
            self._init_new_project(init_project)
        if not os.path.exists(self.paths["config_file"]):
            msg = ("Could not find the project's config file. Wrong project "
                "path or uninitialized project?")
            raise FWIWException(msg)
        self._read_config_file()
        self.update_folder_structure()

    def _setup_paths(self, root_path):
        """
        Central place to define all paths.
        """
        # Every key containing the string "file" denotes a file, all others
        # should denote directories.
        self.paths = {}
        self.paths["root"] = root_path
        self.paths["config_file"] = os.path.join(root_path,
            "config.xml")
        self.paths["events"] = os.path.join(root_path, "EVENTS")
        self.paths["data"] = os.path.join(root_path, "DATA")
        self.paths["logs"] = os.path.join(root_path, "LOGS")
        self.paths["synthetics"] = os.path.join(root_path, "SYNTHETICS")
        self.paths["stations"] = os.path.join(root_path, "STATIONS")
        # Station subfolders
        self.paths["dataless_seed"] = os.path.join(self.paths["stations"],
            "SEED")
        self.paths["station_xml"] = os.path.join(self.paths["stations"],
            "StationXML")
        self.paths["resp"] = os.path.join(self.paths["stations"],
            "RESP")

    def update_folder_structure(self):
        """
        Updates the folder structure of the project.
        """
        for name, path in self.paths.iteritems():
            if "file" in name or os.path.exists(path):
                continue
            os.makedirs(path)
        events = self.get_event_dict().keys()
        folders = [self.paths["data"], self.paths["synthetics"]]
        for folder in folders:
            for event in events:
                event_folder = os.path.join(folder, event)
                if os.path.exists(event_folder):
                    continue
                os.makedirs(event_folder)

    def _init_new_project(self, project_name):
        """
        Initializes a new project. This currently just means that it creates a
        default config file. The folder structure is checked and rebuilt every
        time the project is initialized anyways.
        """
        if not project_name:
            project_name = "FWIWProject"

        doc = E.fwiw_project(
            E.name(project_name),
            E.description(""),
            E.download_settings(
                E.arclink_username(""),
                E.seconds_before_event(str(300)),
                E.seconds_after_event(str(300))),
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

    def __str__(self):
        """
        Pretty string representation. Currently very basic.
        """
        ret_str = "FWIW project \"%s\"\n" % self.config["name"]
        ret_str += "\tDescription: %s\n" % self.config["description"]
        ret_str += "\tProject root: %s\n" % self.paths["root"]
        return ret_str

    def get_station_filename(self, network, station, location, channel,
            format):
        """
        Function returning the filename a station file of a certain format
        should be written to. Only useful as a callback function.

        :type format: String
        :param format: 'datalessSEED', 'StationXML', or 'RESP'
        """
        if format not in ["datalessSEED", "StationXML", "RESP"]:
            msg = "Unknown format '%s'" % format
            raise ValueError(msg)
        if format == "datalessSEED":
            def seed_filename_generator():
                i = 0
                while True:
                    filename = os.path.join(self.paths["dataless_seed"],
                        "dataless.{network}_{station}".format(network=network,
                        station=station))
                    if i:
                        filename += ".%i" % i
                    i += 1
                    yield filename
            for filename in seed_filename_generator():
                if not os.path.exists(filename):
                    break
            return filename
        else:
            raise NotImplementedError

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
        """
        Plots the simulation domain and the actual physical domain.

        Wrapper around one of the visualization routines.
        """
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
        self.events = obspy.readEvents(os.path.join(self.paths["events"],
            "*%sxml" % os.path.extsep))

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

    def has_station_file(self, waveform_filename):
        """
        Simple function to determine whether or not the station file for a
        given waveform file exist. Will return either the filename or False.

        Naming scheme of the files:
            dataless SEED:
                dataless.NETWORK_STATION[.X]
            StationXML:
                station.NETWORK_STATION[.X].xml
            RESP Files:
                RESP.NETWORK.STATION.LOCATION.CHANNEL[.X]

        The [.X] are where a potential number would be appended in the case of
        more then one necessary file.
        """
        tr = obspy.read(waveform_filename)[0]
        network = tr.stats.network
        station = tr.stats.station
        location = tr.stats.location
        channel = tr.stats.channel
        starttime = tr.stats.starttime
        endtime = tr.stats.endtime

        # Check for dataless SEED first. Two step globbing because of limited
        # wildcard capabilities and possibility of false positives otherwise.
        dataless_seed = glob.glob(os.path.join(self.paths["dataless_seed"],
            "dataless.{network}_{station}".format(network=network,
            station=station)))
        dataless_seed.extend(glob.glob(os.path.join(
            self.paths["dataless_seed"],
            "dataless.{network}_{station}.*".format(network=network,
            station=station))))

        for filename in dataless_seed:
            p = Parser(filename)
            import pprint
            pprint.pprint(p.getInventory())
            channels = p.getInventory()["channels"]
            for chan in channels:
                chan_id = "%s.%s.%s.%s" % (network, station, location, channel)
                if chan["channel_id"] != chan_id:
                    continue
                if starttime <= chan["start_date"]:
                    continue
                if chan["end_date"] and \
                        (endtime >= chan["end_date"]):
                    continue
                return filename

        # XXX: Deal with StationXML and RESP files as well!
        return False
