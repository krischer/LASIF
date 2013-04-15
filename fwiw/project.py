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
import copy
from datetime import datetime
import glob
from lxml import etree
from lxml.builder import E
import obspy
from obspy.core.util import FlinnEngdahl
from obspy.xseed import Parser
import os
import matplotlib.pyplot as plt
import sys
from wfs_input_generator import InputFileGenerator
from fwiw import utils, visualization


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
        self.paths["models"] = os.path.join(root_path, "MODELS")
        self.paths["synthetics"] = os.path.join(root_path, "SYNTHETICS")
        self.paths["templates"] = os.path.join(root_path, "TEMPLATES")
        self.paths["source_time_functions"] = os.path.join(root_path,
            "SOURCE_TIME_FUNCTIONS")
        self.paths["stations"] = os.path.join(root_path, "STATIONS")
        # Station subfolders
        self.paths["dataless_seed"] = os.path.join(self.paths["stations"],
            "SEED")
        self.paths["station_xml"] = os.path.join(self.paths["stations"],
            "StationXML")
        self.paths["resp"] = os.path.join(self.paths["stations"],
            "RESP")
        self.paths["output"] = os.path.join(root_path, "OUTPUT")

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

        # Also create one source time function example file.
        stf = (
            "import obspy\n"
            "import numpy as np\n"
            "\n"
            "\n"
            "def filtered_heaviside(npts, delta, freqmin, freqmax):\n"
            "    trace = obspy.Trace(data=np.ones(npts))\n"
            "    trace.stats.delta = delta\n"
            "    trace.filter(\"lowpass\", freq=freqmax, corners=5)\n"
            "    trace.filter(\"highpass\", freq=freqmin, corners=2)\n"
            "\n"
            "    return trace.data\n"
            "\n"
            "\n"
            "def source_time_function(npts, delta):\n"
            "    return filtered_heaviside(npts, delta, 1. / 500., 1. / 60.)")
        # The source time functions path needs to exist.
        if not os.path.exists(self.paths["source_time_functions"]):
            os.makedirs(self.paths["source_time_functions"])
        with open(os.path.join(self.paths["source_time_functions"],
                "heaviside_60s_500s.py"), "wt") as open_file:
            open_file.write(stf)

    def _get_source_time_function(self, function_name):
        """
        Attempts to get the source time function with the corresponding name.

        Will raise if something does not work.
        """
        filename = os.path.join(self.paths["source_time_functions"], "%s.py"
            % function_name)
        if not os.path.exists(filename):
            msg = "Could not find source time function '%s'" % function_name
            raise ValueError(msg)

        # Attempt to import the file if found.
        old_path = copy.copy(sys.path)
        sys.path.insert(1, os.path.dirname(filename))
        try:
            sft = __import__(os.path.splitext(os.path.basename(filename))[0],
                globals(), locals())
        except Exception as e:
            msg = "Could not import '%s'\n" % filename
            msg += "\t%s" % str(e)
            raise Exception(msg)
        finally:
            sys.path = old_path

        if not hasattr(sft, "source_time_function") or \
                not hasattr(sft.source_time_function, "__call__"):
            msg = ("File '%s' does not contain a function "
                "'source_time_function'.") % filename
            raise Exception(msg)
        return sft.source_time_function

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

    def get_model_dict(self):
        """
        Returns a dictonary with all models in the project, the keys are the
        model names and the values the full paths to each model.
        """
        contents = [os.path.join(self.paths["models"], _i)
            for _i in os.listdir(self.paths["models"])]
        models = [os.path.abspath(_i) for _i in contents if os.path.isdir(_i)]
        models = {os.path.basename(_i): _i for _i in models}
        return models

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

    def plot_event(self, event_name, resolution="c"):
        """
        Plots information about one event on the map.
        """
        # Plot the domain.
        bounds = self.domain["bounds"]
        map = visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], bounds["boundary_width_in_degree"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"],
            plot_simulation_domain=False, show_plot=False)

        all_events = self.get_event_dict()
        if event_name not in all_events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)
        event = obspy.readEvents(all_events[event_name])

        stations = self.get_stations_for_event(event_name)
        ev_lng = event[0].preferred_origin().longitude
        ev_lat = event[0].preferred_origin().latitude
        visualization.plot_stations_for_event(map_object=map,
            station_dict=stations, event_longitude=ev_lng,
            event_latitude=ev_lat)
        # Plot the beachball for one event.
        visualization.plot_events(event, map_object=map)

        plt.show()

    def plot_events(self, resolution="c"):
        """
        Plots the domain and beachballs for all events on the map.
        """
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

    def get_event_info(self, event_name):
        """
        Returns a dictionary with information about one, specific event.
        """
        all_events = self.get_event_dict()
        if event_name not in all_events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)
        event = obspy.readEvents(all_events[event_name])[0]
        mag = event.preferred_magnitude()
        org = event.preferred_origin()
        info = {
            "latitude": org.latitude,
            "longitude": org.longitude,
            "origin_time": org.time,
            "depth_in_km": org.depth / 1000.0,
            "magnitude": mag.mag,
            "region": FlinnEngdahl().get_region(org.longitude, org.latitude),
            "magnitude_type": mag.magnitude_type}
        return info

    def generate_input_files(self, event_name, template_name, simulation_type,
            source_time_function):
        """
        Generate the input files for one event.

        :param event_name: The name of the event for which to generate the
            input files.
        :param template_name: The name of the input file template
        :param simulation_type: The type of simulation to perform. Possible
            values are: 'normal simulation', 'adjoint forward', 'adjoint
            reverse'
        :param source_time_function: A function source_time_function(npts,
            delta), taking the requested number of samples and the time spacing
            and returning an appropriate source time function as numpy array.
        """
        # Get the events
        all_events = self.get_event_dict()
        if event_name not in all_events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)

        event = obspy.readEvents(all_events[event_name])[0]

        # Get the input file templates.
        template_filename = os.path.join(self.paths["templates"],
            template_name + ".xml")
        if not os.path.exists(template_filename):
            msg = "Template '%s' does not exists." % template_name
            raise ValueError(msg)
        input_file = utils.read_ses3d_4_0_template(template_filename)

        # Get all stations and create a dictionary for the input file
        # generator.
        stations = self.get_stations_for_event(event_name)
        stations = [{"id": key, "latitude": value["latitude"],
            "longitude": value["longitude"],
            "elevation_in_m": value["elevation"],
            "local_depth_in_m": value["local_depth"]} for key, value in
            stations.iteritems()]

        # Add the event and the stations to the input file generator.
        gen = InputFileGenerator()
        gen.add_events(event)
        gen.add_stations(stations)

        npts = input_file["simulation_parameters"]["number_of_time_steps"]
        delta = input_file["simulation_parameters"]["time_increment"]
        # Time configuration.
        gen.config.number_of_time_steps = npts
        gen.config.time_increment_in_s = delta

        # SES3D specific configuration
        gen.config.output_folder = input_file["output_directory"]
        gen.config.simulation_type = simulation_type

        gen.config.adjoint_forward_wavefield_output_folder = \
            input_file["adjoint_output_parameters"][
                "forward_field_output_directory"]
        gen.config.adjoint_forward_sampling_rate = \
            input_file["adjoint_output_parameters"][
                "sampling_rate_of_forward_field"]
        gen.config.is_dissipative = \
            input_file["simulation_parameters"]["is_dissipative"]

        # Discretization
        disc = input_file["computational_setup"]
        gen.config.nx_global = disc["nx_global"]
        gen.config.ny_global = disc["ny_global"]
        gen.config.nz_global = disc["nz_global"]
        gen.config.px = disc["px_processors_in_theta_direction"]
        gen.config.py = disc["py_processors_in_phi_direction"]
        gen.config.pz = disc["pz_processors_in_r_direction"]
        gen.config.lagrange_polynomial_degree = \
            disc["lagrange_polynomial_degree"]

        # Configure the mesh.
        gen.config.mesh_min_latitude = \
            self.domain["bounds"]["minimum_latitude"]
        gen.config.mesh_max_latitude = \
            self.domain["bounds"]["maximum_latitude"]
        gen.config.mesh_min_longitude = \
            self.domain["bounds"]["minimum_longitude"]
        gen.config.mesh_max_longitude = \
            self.domain["bounds"]["maximum_longitude"]
        gen.config.mesh_min_depth_in_km = \
            self.domain["bounds"]["minimum_depth_in_km"]
        gen.config.mesh_max_depth_in_km = \
            self.domain["bounds"]["maximum_depth_in_km"]

        gen.config.rotation_angle_in_degree = self.domain["rotation_angle"]
        gen.config.rotation_axis = self.domain["rotation_axis"]

        gen.config.source_time_function = source_time_function(int(npts),
            float(delta))

        # Generate the output directory.
        output_dir = "input_files___%s___%s" % (template_name,
            str(datetime.now()).replace(" ", "T"))

        output_dir = os.path.join(self.paths["output"], output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        gen.write(format="ses3d_4_0", output_dir=output_dir)
        print "Written files to '%s'." % output_dir

    def get_stations_for_event(self, event_name):
        """
        Returns a dictionary containing a little bit of information about all
        stations available for a given event.

        An available station is defined as a station with an existing station
        file and an existing waveform file.

        Will return an empty dictionary if nothing is found.
        """
        all_events = self.get_event_dict()
        if event_name not in all_events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)

        data_path = os.path.join(self.paths["data"], event_name, "raw")
        if not os.path.exists(data_path):
            return {}

        station_info = {}

        for waveform_file in glob.iglob(os.path.join(data_path, "*")):
            try:
                tr = obspy.read(waveform_file)[0]
            except:
                continue

            station_id = "%s.%s" % (tr.stats.network, tr.stats.station)
            if station_id in station_info:
                continue

            station_file = self.has_station_file(tr)
            if not station_file:
                continue

            p = Parser(station_file)
            coordinates = p.getCoordinates(tr.id, tr.stats.starttime)
            station_info[station_id] = coordinates

        return station_info

    def has_station_file(self, waveform_filename_or_trace):
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
        if isinstance(waveform_filename_or_trace, basestring):
            tr = obspy.read(waveform_filename_or_trace)[0]
        else:
            tr = waveform_filename_or_trace
        network = tr.stats.network
        station = tr.stats.station
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
            if utils.channel_in_parser(p, tr.id, starttime, endtime) \
                    is True:
                return filename

        # XXX: Deal with StationXML and RESP files as well!
        return False
