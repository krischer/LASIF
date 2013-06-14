#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project management class.

It is important to not import necessary things at the method level to make
importing this file as fast as possible. Otherwise using the command line
interface feels sluggish and slow. Import things only the functions they are
needed.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import cPickle
import glob
import os
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
            plot_simulation_domain=True, show_plot=True, zoom=True)

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
        from lasif import iteration_xml

        iteration_name = iteration_name.replace(" ", "_").upper()
        filename = "ITERATION_%s.xml" % iteration_name
        filename = os.path.join(self.paths["iterations"], filename)
        if os.path.exists(filename):
            msg = "Iteration already exists."
            raise LASIFException(msg)

        # Get a dictionary containing the event names as keys and a list of
        # stations per event as values.
        events_dict = {event: self.get_stations_for_event(event).keys()
            for event in self.get_event_dict().keys()}

        xml_string = iteration_xml.create_iteration_xml_string(
            iteration_name, solver_name, events_dict)

        with open(filename, "wt") as fh:
            fh.write(xml_string)

        print "Created iteration %s" % iteration_name

    def get_iteration_dict(self):
        """
        Returns a dictonary with all iterations in the project, the keys are
        the iteration names and the values the full paths to each iteration.
        """
        iterations = {}
        for iteration in glob.iglob(os.path.join(self.paths["iterations"],
                "*%sxml" % os.extsep)):
            iteration = os.path.abspath(iteration)
            iteration_name = os.path.splitext(os.path.basename(iteration))[0]
            if iteration_name.startswith("ITERATION_"):
                iteration_name = iteration_name[10:]
            iterations[iteration_name] = iteration
        return iterations

    def _get_iteration(self, iteration_name):
        """
        Helper method to read a certain iteration.
        """
        from lasif.iteration_xml import Iteration

        iterations = self.get_iteration_dict()
        return Iteration(iterations[iteration_name])

    def preprocess_data(self, iteration_name):
        """
        Preprocesses all data for a given iteration.
        """
        from lasif import preprocessing
        import colorama
        import obspy

        iteration = self._get_iteration(iteration_name)

        process_params = iteration.get_process_params()
        processing_tag = iteration.get_processing_tag()

        def processing_data_generator():
            for event_name, event in iteration.events.iteritems():
                event_info = self.get_event_info(event_name)
                # The folder where all preprocessed data for this event will
                # go.
                event_data_path = os.path.join(self.paths["data"], event_name,
                    processing_tag)
                if not os.path.exists(event_data_path):
                    os.makedirs(event_data_path)
                # All stations that will be processed for this iteration and
                # event.
                stations = event["stations"].keys()
                waveforms = self._get_waveform_cache_file(event_name, "raw")\
                    .get_values()
                for waveform in waveforms:
                    station_id = "{network}.{station}".format(**waveform)
                    # Only process data from stations needed for the current
                    # iteration.
                    if station_id not in stations:
                        continue
                    # Generate the new filename for the waveform. If it already
                    # exists, continue.
                    processed_filename = os.path.join(event_data_path,
                        os.path.basename(waveform["filename"]))
                    if os.path.exists(processed_filename):
                        continue
                    ret_dict = process_params.copy()
                    ret_dict["data_path"] = waveform["filename"]
                    ret_dict["processed_data_path"] = processed_filename
                    ret_dict.update(event_info)
                    ret_dict["station_filename"] = \
                        self.station_cache.get_station_filename(
                            waveform["channel_id"],
                            obspy.UTCDateTime(waveform["starttime_timestamp"]))
                    yield ret_dict

        count = preprocessing.launch_processing(processing_data_generator())

        print colorama.Fore.GREEN + ("\nDONE - Preprocessed %i files." %
            (count)) + colorama.Style.RESET_ALL

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

    def generate_input_files(self, iteration_name, event_name,
            simulation_type):
        """
        Generate the input files for one event.

        :param iteration_name: The name of the iteration.
        :param event_name: The name of the event for which to generate the
            input files.
        :param simulation_type: The type of simulation to perform. Possible
            values are: 'normal simulation', 'adjoint forward', 'adjoint
            reverse'
        """
        from wfs_input_generator import InputFileGenerator

        iteration = self._get_iteration(iteration_name)
        # Check that the event is part of the iterations.
        if event_name not in iteration.events:
            msg = "Event '%s' not part of iteration '%s'." % (event_name,
                    iteration_name)
            raise ValueError(msg)
        event = self.get_event(event_name)
        stations_for_event = iteration.events[event_name]["stations"].keys()

        # Get all stations and create a dictionary for the input file
        # generator.
        stations = self.get_stations_for_event(event_name)
        stations = [{"id": key, "latitude": value["latitude"],
            "longitude": value["longitude"],
            "elevation_in_m": value["elevation"],
            "local_depth_in_m": value["local_depth"]} for key, value in
            stations.iteritems() if key in stations_for_event]

        solver = iteration.solver_settings

        # Currently only SES3D 4.0 is supported
        if solver["solver"].lower() != "ses3d 4.0":
            msg = "Currently only SES3D 4.0 is supported."
            raise ValueError(msg)

        solver = solver["solver_settings"]

        # Add the event and the stations to the input file generator.
        gen = InputFileGenerator()
        gen.add_events(event)
        gen.add_stations(stations)

        npts = solver["simulation_parameters"]["number_of_time_steps"]
        delta = solver["simulation_parameters"]["time_increment"]
        # Time configuration.
        gen.config.number_of_time_steps = npts
        gen.config.time_increment_in_s = delta

        # SES3D specific configuration
        gen.config.output_folder = solver["output_directory"].replace(
            "{{EVENT_NAME}}", event_name.replace(" ", "_"))
        gen.config.simulation_type = simulation_type

        gen.config.adjoint_forward_wavefield_output_folder = \
            solver["adjoint_output_parameters"][
                "forward_field_output_directory"].replace(
                    "{{EVENT_NAME}}", event_name.replace(" ", "_"))
        gen.config.adjoint_forward_sampling_rate = \
            solver["adjoint_output_parameters"][
                "sampling_rate_of_forward_field"]

        diss = solver["simulation_parameters"]["is_dissipative"]
        if diss.lower() == "false":
            diss = False
        elif diss.lower() == "true":
            diss = True
        else:
            msg = ("is_dissipative value of '%s' unknown. Choose "
                "true or false.") % diss
            raise ValueError(msg)
        gen.config.is_dissipative = diss

        # Discretization
        disc = solver["computational_setup"]
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
        # Set the rotation parameters.
        gen.config.rotation_angle_in_degree = self.domain["rotation_angle"]
        gen.config.rotation_axis = self.domain["rotation_axis"]

        gen.config.source_time_function = \
            iteration.get_source_time_function()["data"]

        output_dir = self.get_output_folder(
            "input_files___ITERATION_%s__EVENT_%s" % (iteration_name,
            event_name))

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

    def finalize_adjoint_sources(self, iteration_name, event_name):
        """
        Finalizes the adjoint sources.
        """
        from itertools import izip
        import numpy as np

        from lasif import rotations
        from lasif.window_manager import MisfitWindowManager
        from lasif.adjoint_src_manager import AdjointSourceManager

        iteration = self._get_iteration(iteration_name)
        long_iteration_name = "ITERATION_%s" % iteration_name

        window_directory = os.path.join(self.paths["windows"], event_name,
            long_iteration_name)
        ad_src_directory = os.path.join(self.paths["adjoint_sources"],
            event_name, long_iteration_name)
        window_manager = MisfitWindowManager(window_directory,
            long_iteration_name, event_name)
        adj_src_manager = AdjointSourceManager(ad_src_directory)

        this_event = iteration.events[event_name]

        event_weight = this_event["event_weight"]
        all_stations = self.get_stations_for_event(event_name)

        all_coordinates = []
        _i = 0

        output_folder = self.get_output_folder(
            "adjoint_sources__ITERATION_%s__%s" % (iteration_name, event_name))

        for station_name, station in this_event["stations"].iteritems():
            this_station = all_stations[station_name]

            station_weight = station["station_weight"]
            windows = window_manager.get_windows_for_station(station_name)
            if not windows:
                msg = "No adjoint sources for station '%s'." % station_name
                warnings.warn(msg)
                continue

            all_channels = {}

            for channel_windows in windows:
                channel_id = channel_windows["channel_id"]
                cumulative_weight = 0
                all_data = []
                for window in channel_windows["windows"]:
                    window_weight = window["weight"]
                    starttime = window["starttime"]
                    endtime = window["endtime"]
                    data = adj_src_manager.get_adjoint_src(channel_id,
                        starttime, endtime)
                    all_data.append(window_weight * data)
                    cumulative_weight += window_weight
                data = all_data.pop()
                for d in all_data:
                    data += d
                data /= cumulative_weight
                data *= station_weight * event_weight
                all_channels[channel_id[-1]] = data

            length = len(all_channels.values()[0])
            # Use zero for empty ones.
            for component in ["N", "E", "Z"]:
                if component in all_channels:
                    continue
                all_channels[component] = np.zeros(length)

            # Rotate.
            rec_lat = this_station["latitude"]
            rec_lng = this_station["longitude"]
            if self.domain["rotation_angle"]:
                # Rotate the adjoint source location.
                r_rec_lat, r_rec_lng = rotations.rotate_lat_lon(rec_lat,
                    rec_lng, self.domain["rotation_axis"],
                    -self.domain["rotation_angle"])
                # Rotate the data.
                all_channels["N"], all_channels["E"], all_channels["Z"] = \
                    rotations.rotate_data(all_channels["N"], all_channels["E"],
                        all_channels["Z"], rec_lat, rec_lng,
                        self.domain["rotation_axis"], self.domain["angle"])
            else:
                r_rec_lat = rec_lat
                r_rec_lng = rec_lng
            r_rec_depth = 0.0
            r_rec_colat = rotations.lat2colat(r_rec_lat)

            CHANNEL_MAPPING = {"X": "N", "Y": "E", "Z": "Z"}

            _i += 1

            adjoint_src_filename = os.path.join(output_folder,
                "ad_src_%i" % _i)

            all_coordinates.append((r_rec_colat, r_rec_lng, r_rec_depth))

            # Actually write the adjoint source file in SES3D specific format.
            with open(adjoint_src_filename, "wt") as open_file:
                open_file.write("-- adjoint source ------------------\n")
                open_file.write("-- source coordinates (colat,lon,depth)\n")
                open_file.write("%f %f %f\n" % (r_rec_colat, r_rec_lng,
                    r_rec_depth))
                open_file.write("-- source time function (x, y, z) --\n")
                for x, y, z in izip(
                        -1.0 * all_channels[CHANNEL_MAPPING["X"]],
                        all_channels[CHANNEL_MAPPING["Y"]],
                        -1.0 * all_channels[CHANNEL_MAPPING["Z"]]):
                    open_file.write("%e %e %e\n" % (x, y, z))
                open_file.write("\n")

        # Write the final file.
        with open(os.path.join(output_folder, "ad_srcfile"), "wt") as fh:
            fh.write("%i\n" % _i)
            for line in all_coordinates:
                fh.write("%.6f %.6f %.6f\n" % (line[0], line[1], line[2]))
            fh.write("\n")

        print "Wrote %i adjoint sources to %s." % (_i, output_folder)

    def data_synthetic_iterator(self, event_name, iteration_name):
        from lasif import rotations
        from obspy import read, Stream

        event_info = self.get_event_info(event_name)
        iteration = self._get_iteration(iteration_name)
        iteration_stations = iteration.events[event_name]["stations"].keys()

        stations = {key: value for key, value in
            self.get_stations_for_event(event_name).iteritems() if key in
            iteration_stations}

        waveforms = \
            self._get_waveform_cache_file(event_name,
                iteration.get_processing_tag()).get_values()

        long_iteration_name = "ITERATION_%s" % iteration_name
        synthetics_path = os.path.join(self.paths["synthetics"], event_name,
            long_iteration_name)
        synthetic_files = {os.path.basename(_i).replace("_", ""): _i for _i in
            glob.iglob(os.path.join(synthetics_path, "*"))}

        if not synthetic_files:
            msg = "Could not find any synthetic files in '%s'." % \
                synthetics_path
            raise ValueError(msg)

        SYNTH_MAPPING = {"X": "N", "Y": "E", "Z": "Z"}

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

                for key, value in this_waveforms.iteritems():
                    data += read(value["filename"])[0]
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

                # Scale the data
                try:
                    n_d_trace = data.select(component="N")[0]
                except:
                    n_d_trace = None
                try:
                    e_d_trace = data.select(component="E")[0]
                except:
                    e_d_trace = None
                try:
                    z_d_trace = data.select(component="Z")[0]
                except:
                    z_d_trace = None
                n_s_trace = synthetics.select(component="N")[0]
                e_s_trace = synthetics.select(component="E")[0]
                z_s_trace = synthetics.select(component="Z")[0]

                # Rotate the synthetics if nessesary.
                if self.rot_angle:
                    # First rotate the station back to see, where it was
                    # recorded.
                    lat, lng = rotations.rotate_lat_lon(
                        coordinates["latitude"], coordinates["longitude"],
                        self.rot_axis, -self.rot_angle)
                    # Rotate the data.
                    n, e, z = rotations.rotate_data(n_s_trace.data,
                        e_s_trace.data, z_s_trace.data, lat, lng,
                        self.rot_axis, self.rot_angle)
                    n_s_trace.data = n
                    e_s_trace.data = e
                    z_s_trace.data = z

                # Scale the data to the synthetics.
                if n_d_trace:
                    scaling_factor = n_s_trace.data.ptp() / \
                        n_d_trace.data.ptp()
                    n_d_trace.stats.scaling_factor = scaling_factor
                    n_d_trace.data *= scaling_factor
                if e_d_trace:
                    scaling_factor = e_s_trace.data.ptp() / \
                        e_d_trace.data.ptp()
                    e_d_trace.stats.scaling_factor = scaling_factor
                    e_d_trace.data *= scaling_factor
                if z_d_trace:
                    scaling_factor = z_s_trace.data.ptp() / \
                        z_d_trace.data.ptp()
                    z_d_trace.stats.scaling_factor = scaling_factor
                    z_d_trace.data *= scaling_factor

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
