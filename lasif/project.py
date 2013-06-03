#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project management class.

It is important to import necessary things at the method level to make
importing this file as fast as possible. Otherwise using the command line
interface feels sluggish and slow.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import cPickle
import glob
import os
import sys
import warnings


class LASIFException(Exception):
    pass


class Project(object):
    """
    A class representing and managing a single LASIF project.
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
            raise LASIFException(msg)
        self.update_folder_structure()
        self._read_config_file()

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
        self.paths["cache"] = os.path.join(root_path, "CACHE")
        self.paths["config_file_cache"] = os.path.join(self.paths["cache"],
            "config.xml_cache.pickle")
        self.paths["inv_db_file"] = os.path.join(self.paths["cache"],
            "inventory_db.sqlite")
        self.paths["logs"] = os.path.join(root_path, "LOGS")
        self.paths["models"] = os.path.join(root_path, "MODELS")
        self.paths["iterations"] = os.path.join(root_path, "ITERATIONS")
        self.paths["synthetics"] = os.path.join(root_path, "SYNTHETICS")
        self.paths["templates"] = os.path.join(root_path, "TEMPLATES")
        self.paths["processing_functions"] = os.path.join(root_path,
            "PROCESSING_FUNCTIONS")
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
        self.paths["windows"] = os.path.join(root_path,
            "ADJOINT_SOURCES_AND_WINDOWS", "WINDOWS")
        self.paths["adjoint_sources"] = os.path.join(root_path,
            "ADJOINT_SOURCES_AND_WINDOWS", "ADJOINT_SOURCES")

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
        import copy
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
        ret_str = "LASIF project \"%s\"\n" % self.config["name"]
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
        if format == "RESP":
            def resp_filename_generator():
                i = 0
                while True:
                    filename = os.path.join(self.paths["resp"],
                        "RESP.{network}.{station}.{location}.{channel}"
                        .format(network=network, station=station,
                            location=location, channel=channel))
                    if i:
                        filename += ".%i" % i
                    i += 1
                    yield filename
            for filename in resp_filename_generator():
                if not os.path.exists(filename):
                    break
            return filename
        else:
            raise NotImplementedError

    def _read_config_file(self):
        """
        Parse the config file.
        """
        # Attempt to read the cached config file. This might seem excessive but
        # since this file is read every single time a LASIF command is used it
        # makes difference at least in the perceived speed of LASIF.
        cfile = self.paths["config_file_cache"]
        if os.path.exists(cfile):
            with open(cfile, "rb") as fh:
                cf_cache = cPickle.load(fh)
            last_m_time = int(os.path.getmtime(self.paths["config_file"]))
            if last_m_time == cf_cache["last_m_time"]:
                self.config = cf_cache["config"]
                self.domain = cf_cache["domain"]
                return

        from lxml import etree
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

        # Write cache file.
        cf_cache = {}
        cf_cache["config"] = self.config
        cf_cache["domain"] = self.domain
        cf_cache["last_m_time"] = \
            int(os.path.getmtime(self.paths["config_file"]))
        with open(cfile, "wb") as fh:
            cPickle.dump(cf_cache, fh, protocol=2)

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
        from lasif import visualization

        bounds = self.domain["bounds"]
        visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], bounds["boundary_width_in_degree"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"],
            plot_simulation_domain=True, show_plot=True)

    def get_event(self, event_name):
        """
        Helper function to avoid reading one event twice.
        """
        from obspy import readEvents

        if not hasattr(self, "_seismic_events"):
            self._seismic_events = {}
        # Read the file if it does not exist.
        if event_name not in self._seismic_events:
            filename = os.path.join(self.paths["events"], "%s%sxml" %
                (event_name, os.path.extsep))
            if not os.path.exists(filename):
                return None
            self._seismic_events[event_name] = readEvents(filename)[0]
        return self._seismic_events[event_name]

    def create_new_iteration(self, iteration_name, solver_name):
        """
        Creates a new iteration file.
        """
        from lxml import etree
        from lxml.builder import E

        iteration_name = iteration_name.replace(" ", "_").upper()
        filename = "ITERATION_%s.xml" % iteration_name
        filename = os.path.join(self.paths["iterations"], filename)
        if os.path.exists(filename):
            msg = "Iteration already exists."
            raise LASIFException(msg)

        solver_doc = self._get_default_solver_settings(solver_name)
        if solver_name.lower() == "ses3d_4_0":
            solver_name = "SES3D 4.0"
        else:
            raise NotImplementedError

        # Loop over all events.
        events = self.get_event_dict().keys()
        events_doc = [E.event(E.event_name(_i), E.event_weight("1.0"),
            E.time_correction("0.0")) for _i in events]

        stations = {}
        events_doc = []
        # Also over all stations.
        for event in events:
            stations = self.get_stations_for_event(event)
            stations_doc = [E.station(
                E.station_id(station),
                E.station_weight("1.0"),
                E.time_correction_in_s("0.0")) for station in stations]
            events_doc.append(E.event(
                E.event_name(event),
                E.event_weight("1.0"),
                E.time_correction_in_s("0.0"),
                *stations_doc))

        doc = E.iteration(
            E.iteration_name(iteration_name),
            E.iteration_description(""),
            E.comment(""),
            E.data_preprocessing(
                E.highpass_frequency("0.01"),
                E.lowpass_frequency("0.5")),
            E.rejection_criteria(""),
            E.source_time_function("Filtered Heaviside"),
            E.solver_parameters(
                E.solver(solver_name),
                solver_doc),
            *events_doc)

        string_doc = etree.tostring(doc, pretty_print=True,
            xml_declaration=True, encoding="UTF-8")
        with open(filename, "wt") as fh:
            fh.write(string_doc)

        print "Created iteration %s" % iteration_name

    def _get_default_solver_settings(self, solver):
        """
        Helper function returning etree representation of a solver's default
        settings.
        """
        known_solvers = ["ses3d_4_0A"]
        if solver.lower() == "ses3d_4_0":
            from lasif.utils import generate_ses3d_4_0_template
            return generate_ses3d_4_0_template()
        else:
            msg = "Solver '%s' not known. Known solvers: %s" % (solver,
                ",".join(known_solvers))
            raise LASIFException(msg)

    def get_all_events(self):
        """
        Parses all events and returns a list of Event objects.
        """
        events = self.get_event_dict()
        for event in events.keys():
            self.get_event(event)
        return self._seismic_events.values()

    def plot_event(self, event_name):
        """
        Plots information about one event on the map.
        """
        from lasif import visualization
        import matplotlib.pyplot as plt

        # Plot the domain.
        bounds = self.domain["bounds"]
        map = visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], bounds["boundary_width_in_degree"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"],
            plot_simulation_domain=False, show_plot=False, zoom=True)

        all_events = self.get_event_dict()
        if event_name not in all_events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)

        event = self.get_event(event_name)
        event_info = self.get_event_info(event_name)

        stations = self.get_stations_for_event(event_name)
        visualization.plot_stations_for_event(map_object=map,
            station_dict=stations, event_info=event_info)
        # Plot the beachball for one event.
        visualization.plot_events([event], map_object=map)

        plt.show()

    def plot_events(self):
        """
        Plots the domain and beachballs for all events on the map.
        """
        from lasif import visualization
        import matplotlib.pyplot as plt

        bounds = self.domain["bounds"]
        map = visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], bounds["boundary_width_in_degree"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"],
            plot_simulation_domain=False, show_plot=False, zoom=True)
        events = self.get_all_events()
        visualization.plot_events(events, map_object=map)
        plt.show()

    def plot_raydensity(self):
        """
        Plots the raydensity.
        """
        from lasif import visualization
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 21))

        bounds = self.domain["bounds"]
        map = visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], bounds["boundary_width_in_degree"],
            rotation_axis=self.domain["rotation_axis"],
            rotation_angle_in_degree=self.domain["rotation_angle"],
            plot_simulation_domain=False, show_plot=False, zoom=True,
            resolution="l")

        event_stations = []
        for event_name in self.get_event_dict().keys():
            event = self.get_event(event_name)
            stations = self.get_stations_for_event(event_name)
            event_stations.append((event, stations))

        visualization.plot_raydensity(map, event_stations,
            bounds["minimum_latitude"], bounds["maximum_latitude"],
            bounds["minimum_longitude"], bounds["maximum_longitude"],
            self.domain["rotation_axis"], self.domain["rotation_angle"])

        events = self.get_all_events()
        visualization.plot_events(events, map_object=map)

        plt.tight_layout()

        outfile = os.path.join(self.get_output_folder("raydensity_plot"),
            "raydensity.png")
        plt.savefig(outfile, dpi=200)
        print "Saved picture at %s" % outfile

    def get_event_info(self, event_name):
        """
        Returns a dictionary with information about one, specific event.
        """
        from obspy.core.util import FlinnEngdahl

        all_events = self.get_event_dict()
        if event_name not in all_events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)
        event = self.get_event(event_name)
        mag = event.preferred_magnitude() or event.magnitudes[0]
        org = event.preferred_origin() or event.origins[0]

        if org.depth is None:
            warnings.warn("Origin contains no depth. Will be assumed to be 0")
            org.depth = 0.0

        if mag.magnitude_type is None:
            warnings.warn("Magnitude has no specified type. Will be assumed "
                "to be Mw")
            mag.magnitude_type = "Mw"

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
        from lasif import utils
        from wfs_input_generator import InputFileGenerator

        # Get the events
        all_events = self.get_event_dict()
        if event_name not in all_events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)

        event = self.get_event(event_name)

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

        output_dir = self.get_output_folder("input_files___%s" % template_name)

        gen.write(format="ses3d_4_0", output_dir=output_dir)
        print "Written files to '%s'." % output_dir

    def get_output_folder(self, tag):
        """
        Generates a output folder in a unified way.
        """
        from datetime import datetime
        output_dir = ("%s___%s" % (str(datetime.now()), tag)).replace(" ", "T")
        output_dir = os.path.join(self.paths["output"], output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    @property
    def station_cache(self):
        """
        Kind of like an instance wide StationCache singleton.
        """
        from lasif.tools.station_cache import StationCache
        if hasattr(self, "_station_cache"):
            return self._station_cache
        self._station_cache = StationCache(os.path.join(self.paths["cache"],
            "station_cache.sqlite"), self.paths["dataless_seed"],
            self.paths["resp"])
        return self._station_cache

    @station_cache.setter
    def station_cache(self, value):
        msg = "Not allowed. Please update the StationCache instance instead."
        raise Exception(msg)

    def _get_waveform_cache_file(self, event_name, tag):
        """
        Helper function returning the waveform cache file for the data from a
        specific event and a certain tag.
        Example to return the cache for the original data for 'event_1':
        _get_waveform_cache_file("event_1", "raw")
        """
        from lasif.tools.waveform_cache import WaveformCache

        waveform_db_file = os.path.join(self.paths["data"], event_name,
            "%s_cache.sqlite" % tag)
        data_path = os.path.join(self.paths["data"], event_name, tag)
        return WaveformCache(waveform_db_file, data_path)

    def get_stations_for_event(self, event_name):
        """
        Returns a dictionary containing a little bit of information about all
        stations available for a given event.

        An available station is defined as a station with an existing station
        file and an existing waveform file.

        Will return an empty dictionary if nothing is found.
        """
        from lasif.tools.inventory_db import get_station_coordinates

        all_events = self.get_event_dict()
        if event_name not in all_events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)

        data_path = os.path.join(self.paths["data"], event_name, "raw")
        if not os.path.exists(data_path):
            return {}

        waveforms = self._get_waveform_cache_file(event_name, "raw")

        # Query the station cache for a list of all channels.
        available_channels = self.station_cache.get_channels()
        stations = {}
        for waveform in waveforms.get_values():
            station = "%s.%s" % (waveform["network"], waveform["station"])
            # Do not add if already exists.
            if station in stations:
                continue
            # Check if a corresponding station file exists, otherwise skip.
            chan_id = waveform["channel_id"]
            if chan_id not in available_channels:
                continue
            waveform_channel = available_channels[chan_id][0]
            # Now check if the waveform has coordinates (in the case of SAC
            # files).
            if waveform["latitude"]:
                stations[station] = {
                    "latitude": waveform["latitude"],
                    "longitude": waveform["longitude"],
                    "elevation": waveform["elevation_in_m"],
                    "local_depth": waveform["local_depth_in_m"]}
            elif waveform_channel["latitude"]:
                stations[station] = {
                    "latitude": waveform_channel["latitude"],
                    "longitude": waveform_channel["longitude"],
                    "elevation": waveform_channel["elevation_in_m"],
                    "local_depth": waveform_channel["local_depth_in_m"]}
            else:
                # Now check if the station_coordinates are available in the
                # inventory DB and use those.
                coords = get_station_coordinates(self.paths["inv_db_file"],
                    station)

                if coords:
                    stations[station] = {
                        "latitude": coords["latitude"],
                        "longitude": coords["longitude"],
                        "elevation": coords["elevation_in_m"],
                        "local_depth": coords["local_depth_in_m"]}
                else:
                    msg = "No coordinates available for waveform file '%s'" % \
                        waveform["filename"]
                    warnings.warn(msg)

        return stations

    def data_synthetic_iterator(self, event_name, data_tag, synthetic_tag,
            highpass, lowpass):
        from lasif import rotations
        import numpy as np
        from obspy import read, Stream, UTCDateTime
        from obspy.xseed import Parser
        from scipy.interpolate import interp1d

        event_info = self.get_event_info(event_name)

        stations = self.get_stations_for_event(event_name)
        waveforms = \
            self._get_waveform_cache_file(event_name, data_tag).get_values()

        synthetics_path = os.path.join(self.paths["synthetics"], event_name,
            synthetic_tag)
        synthetic_files = {os.path.basename(_i).replace("_", ""): _i for _i in
            glob.iglob(os.path.join(synthetics_path, "*"))}

        SYNTH_MAPPING = {"X": "N", "Y": "E", "Z": "Z"}

        station_cache = self.station_cache

        class TwoWayIter(object):
            def __init__(self, rot_angle=0.0, rot_axis=[0.0, 0.0, 1.0]):
                self.items = stations.items()
                self.current_index = -1
                self.rot_angle = rot_angle
                self.rot_axis = rot_axis

            def next(self):
                self.current_index += 1
                if self.current_index > (len(self.items) - 1):
                    self.current_index = len(self.items) - 1
                    raise StopIteration
                return self.get_value()

            def prev(self):
                self.current_index -= 1
                if self.current_index < 0:
                    self.current_index = 0
                    raise StopIteration
                return self.get_value()

            def get_value(self):
                station_id, coordinates = self.items[self.current_index]

                data = Stream()
                # Now get the actual waveform files. Also find the
                # corresponding station file and check the coordinates.
                this_waveforms = {_i["channel_id"]: _i for _i in waveforms
                    if _i["channel_id"].startswith(station_id + ".")}
                marked_for_deletion = []
                for key, value in this_waveforms.iteritems():
                    value["trace"] = read(value["filename"])[0]
                    data += value["trace"]
                    value["station_file"] = \
                        station_cache.get_station_filename(
                            value["channel_id"],
                            UTCDateTime(value["starttime_timestamp"]))
                    if value["station_file"] is None:
                        marked_for_deletion.append(key)
                        msg = ("Warning: Data and station information for '%s'"
                               " is available, but the station information "
                               "only for the wrong timestamp. You should try "
                               "and retrieve the correct station file.")
                        warnings.warn(msg % value["channel_id"])
                        continue
                    data[-1].stats.station_file = value["station_file"]
                for key in marked_for_deletion:
                    del this_waveforms[key]
                if not this_waveforms:
                    msg = "Could not retrieve data for station '%s'." % \
                        station_id
                    warnings.warn(msg)
                    return None
                # Now attempt to get the synthetics.
                synthetics_filenames = []
                for name, path in synthetic_files.iteritems():
                    if (station_id + ".") in name:
                        synthetics_filenames.append(path)

                if len(synthetics_filenames) != 3:
                    msg = "Found %i not 3 synthetics for station '%s'." % (
                        len(synthetics_filenames), station_id)
                    warnings.warn(msg)
                    return None

                synthetics = Stream()
                # Read all synthetics.
                for filename in synthetics_filenames:
                    synthetics += read(filename)
                for synth in synthetics:
                    if synth.stats.channel in ["X", "Z"]:
                        synth.data *= -1.0
                    synth.stats.channel = SYNTH_MAPPING[synth.stats.channel]
                    synth.stats.starttime = event_info["origin_time"]

                # Process the data.
                len_synth = synthetics[0].stats.endtime - \
                    synthetics[0].stats.starttime
                data.trim(synthetics[0].stats.starttime - len_synth * 0.05,
                    synthetics[0].stats.endtime + len_synth * 0.05)
                if data:
                    max_length = max([tr.stats.npts for tr in data])
                else:
                    max_length = 0
                if max_length == 0:
                    msg = ("Warning: After trimming the waveform data to "
                        "the time window of the synthetics, no more data is "
                        "left. The reference time is the one given in the "
                        "QuakeML file. Make sure it is correct and that "
                        "the waveform data actually contains data in that "
                        "time span.")
                    warnings.warn(msg)
                data.detrend("linear")
                data.taper()

                new_time_array = np.linspace(
                    synthetics[0].stats.starttime.timestamp,
                    synthetics[0].stats.endtime.timestamp,
                    synthetics[0].stats.npts)

                # Simulate the traces.
                for trace in data:
                    # Decimate in case there is a large difference between
                    # synthetic sampling rate and sampling_rate of the data.
                    # XXX: Ugly filter, change!
                    if trace.stats.sampling_rate > (6 *
                            synth.stats.sampling_rate):
                        new_nyquist = trace.stats.sampling_rate / 2.0 / 5.0
                        trace.filter("lowpass", freq=new_nyquist, corners=4,
                            zerophase=True)
                        trace.decimate(factor=5, no_filter=None)

                    station_file = trace.stats.station_file
                    if "/SEED/" in station_file:
                        paz = Parser(station_file).getPAZ(trace.id,
                            trace.stats.starttime)
                        trace.simulate(paz_remove=paz)
                    elif "/RESP/" in station_file:
                        trace.simulate(seedresp={"filename": station_file,
                            "units": "VEL", "date": trace.stats.starttime})
                    else:
                        raise NotImplementedError

                    # Make sure that the data array is at least as long as the
                    # synthetics array. Also add some buffer sample for the
                    # spline interpolation to work in any case.
                    buf = synth.stats.delta * 5
                    if synth.stats.starttime < (trace.stats.starttime + buf):
                        trace.trim(starttime=synth.stats.starttime - buf,
                            pad=True, fill_value=0.0)
                    if synth.stats.endtime > (trace.stats.endtime - buf):
                        trace.trim(endtime=synth.stats.endtime + buf, pad=True,
                            fill_value=0.0)

                    old_time_array = np.linspace(
                        trace.stats.starttime.timestamp,
                        trace.stats.endtime.timestamp,
                        trace.stats.npts)

                    # Interpolation.
                    trace.data = interp1d(old_time_array, trace.data,
                        kind=1)(new_time_array)
                    trace.stats.starttime = synthetics[0].stats.starttime
                    trace.stats.sampling_rate = \
                        synthetics[0].stats.sampling_rate

                data.filter("bandpass", freqmin=lowpass, freqmax=highpass)
                synthetics.filter("bandpass", freqmin=lowpass,
                    freqmax=highpass)

                # Rotate the synthetics if nessesary.
                if self.rot_angle:
                    # First rotate the station back to see, where it was
                    # recorded.
                    lat, lng = rotations.rotate_lat_lon(
                        coordinates["latitude"], coordinates["longitude"],
                        self.rot_axis, -self.rot_angle)
                    # Rotate the data.
                    n_trace = synthetics.select(component="N")[0]
                    e_trace = synthetics.select(component="E")[0]
                    z_trace = synthetics.select(component="Z")[0]
                    n, e, z = rotations.rotate_data(n_trace.data, e_trace.data,
                        z_trace.data, lat, lng, self.rot_axis, self.rot_angle)
                    n_trace.data = n
                    e_trace.data = e
                    z_trace.data = z

                return {"data": data, "synthetics": synthetics,
                    "coordinates": coordinates}

        return TwoWayIter(self.domain["rotation_angle"],
            self.domain["rotation_axis"])

    def has_station_file(self, channel_id, time):
        """
        Simple function returning True or False, if the channel specified with
        it's filename actually has a corresponding station file.
        """
        return self.station_cache.station_info_available(channel_id, time)
