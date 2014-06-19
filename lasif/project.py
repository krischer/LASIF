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
import colorama
import cPickle
import glob
import os
import sys
import warnings

from lasif import LASIFError
from lasif.tools.event_pseudo_dict import EventPseudoDict


# SES3D currently only identifies synthetics  via the filename. Use this
# template to get the name of a certain file.
SYNTHETIC_FILENAME_TEMPLATE = \
    "{network:_<2}.{station:_<5}.{location:_<3}.{component}"


class Project(object):
    """
    A class managing LASIF projects.

    It represents the heart of LASIF.
    """

    def get_station_filename(self, network, station, location, channel,
                             file_format):
        """
        Function returning the filename a station file of a certain format
        should be written to. Only useful as a callback function.

        :type file_format: str
        :param file_format: 'datalessSEED', 'StationXML', or 'RESP'
        """
        if file_format not in ["datalessSEED", "StationXML", "RESP"]:
            msg = "Unknown format '%s'" % file_format
            raise ValueError(msg)
        if file_format == "datalessSEED":
            def seed_filename_generator():
                i = 0
                while True:
                    filename = os.path.join(
                        self.paths["dataless_seed"],
                        "dataless.{network}_{station}".format(
                            network=network, station=station))
                    if i:
                        filename += ".%i" % i
                    i += 1
                    yield filename
            for filename in seed_filename_generator():
                if not os.path.exists(filename):
                    break
            return filename
        if file_format == "RESP":
            def resp_filename_generator():
                i = 0
                while True:
                    filename = os.path.join(
                        self.paths["resp"],
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


    def get_waveform_data(self, event_name, station_id, data_type, tag=None,
                          iteration_name=None):
        """
        Get an ObsPy :class:`~obspy.core.stream.Stream` representing the
        requested waveform data.

        **This is the only interface that should ever be used to read waveform
        data.** This assures a consistent handling of various issues.

        Synthetic data will be properly rotated in case the project is defined
        for a rotated domain.

        :type event_name: str
        :param event_name: The name of the event.
        :type station_id: str
        :param station_id: The id of the station in question.
        :type data_type: str
        :param data_type: The type of data to retrieve. Can be either `raw`,
            `processed`, or `synthetic`.
        :type tag: str
        :param tag: If requesting `processed` data, the processing tag must be
            given.
        :type iteration_name: str
        :param iteration_name: If requesting `synthetic` data, the iteration
            name must be given.

        :rtype: :class:`obspy.core.stream.Stream`
        :return: An up to three-component Stream object containing the
            requested data.
        """
        from lasif import rotations
        import obspy

        # Basic sanity checks.
        if data_type not in ("raw", "processed", "synthetic"):
            msg = "Invalid data_type."
            raise ValueError(msg)
        elif event_name not in self.events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)
        elif data_type == "processed" and tag is None:
            msg = "Tag must be given for requesting processed data."
            raise ValueError(msg)
        elif data_type == "synthetic" and iteration_name is None:
            msg = "iteration_name must be given for requesting synthetic data."
            raise ValueError(msg)

        # Get some station information. This essentially assures the
        # availability of the raw data.
        try:
            self.get_stations_for_event(event_name, station_id)
        except:
            msg = "No raw data found for event '%s' and station '%s'." % (
                event_name, station_id)
            raise LASIFError(msg)

        network, station = station_id.split(".")

        # Diverge to handle the different data types. Raw and processed can be
        # handled almost the same.
        if data_type in ("raw", "processed"):
            waveforms = self._get_waveform_cache_file(event_name, "raw")
            if not waveforms:
                msg = "No suitable data found."
                raise LASIFError(msg)
            waveforms = waveforms.get_files_for_station(network, station)
            # Make sure only one location and channel type component is used.
            channel_set = set()
            for waveform in waveforms:
                channel_set.add(waveform["channel_id"][:-1])
            if len(channel_set) != 1:
                msg = ("More than one combination of location and channel type"
                       " found for event '%s' and station '%s'. The first one "
                       "found will be chosen. Please run the data validation "
                       "to identify and fix the problem.") % (
                    event_name, station_id)
                warnings.warn(msg)
            channel = sorted(list(channel_set))[0]
            filenames = [_i["filename"] for _i in waveforms
                         if waveform["channel_id"].startswith(channel)]

            # If processed, one wants to get the processed files.
            if data_type == "processed":
                filenames = [os.path.join(self.paths["data"], event_name, tag,
                                          os.path.basename(_i))
                             for _i in filenames]
                filenames = [_i for _i in filenames if os.path.exists(_i)]
                if not filenames:
                    msg = ("Failed to find processed files. Did you "
                           "preprocess the data?")
                    raise LASIFError(msg)
            st = obspy.Stream()
            for filename in filenames:
                st += obspy.read(filename)
            return st

        # Synthetics are different.
        elif data_type == "synthetic":
            # First step is to get the folder.
            folder_name = os.path.join(
                self.paths["synthetics"], event_name,
                self._get_long_iteration_name(iteration_name))
            if not os.path.exists(folder_name):
                msg = "Could not find suitable synthetics."
                raise LASIFError(msg)

            # Find all files.
            files = []
            for component in ("X", "Y", "Z"):
                filename = os.path.join(
                    folder_name, SYNTHETIC_FILENAME_TEMPLATE.format(
                        network=network, station=station, location="",
                        component=component.lower()))
                if not os.path.exists(filename):
                    continue
                files.append(filename)

            if not files:
                msg = "Could not find suitable synthetics."
                raise LASIFError(msg)

            # This maps the synthetic channels to ZNE.
            synthetic_coordinates_mapping = {"X": "N", "Y": "E", "Z": "Z"}

            synthetics = obspy.Stream()
            for filename in files:
                tr = obspy.read(filename)[0]
                # Assign network and station codes.
                tr.stats.network = network
                tr.stats.station = station
                # Flip South and downwards pointing data. We want North and Up.
                if tr.stats.channel in ["X"]:
                    tr.data *= -1.0
                tr.stats.channel = \
                    synthetic_coordinates_mapping[tr.stats.channel]
                # Set the correct starttime.
                tr.stats.starttime = \
                    self.events[event_name]["origin_time"]
                synthetics += tr

            # Only three-channel synthetics can be read.
            if sorted([_i.stats.channel for _i in synthetics]) \
                    != ["E", "N", "Z"]:
                msg = ("Could not find all three required components for the "
                       "synthetics for event '%s' at station '%s' for "
                       "iteration '%s'." % (event_name, station_id,
                                            iteration_name))
                raise LASIFError(msg)

            synthetics.sort()

            # Finished if no rotation is required.
            if not self.domain["rotation_angle"]:
                return synthetics

            coordinates = self._get_coordinates_for_waveform_file(
                files[0], "synthetic", network, station, event_name)

            # First rotate the station back to see, where it was
            # recorded.
            lat, lng = rotations.rotate_lat_lon(
                coordinates["latitude"], coordinates["longitude"],
                self.domain["rotation_axis"], -self.domain["rotation_angle"])
            # Rotate the synthetics.
            n, e, z = rotations.rotate_data(
                synthetics.select(channel="N")[0].data,
                synthetics.select(channel="E")[0].data,
                synthetics.select(channel="Z")[0].data,
                lat, lng,
                self.domain["rotation_axis"],
                self.domain["rotation_angle"])
            synthetics.select(channel="N")[0].data = n
            synthetics.select(channel="E")[0].data = e
            synthetics.select(channel="Z")[0].data = z

            return synthetics
        # This should never be reached.
        else:
            raise Exception

    def discover_available_data(self, event_name, station_id):
        """
        Discovers the available data for one event at a certain station.

        Will raise a :exc:`~lasif.project.LASIFError` if no raw data is
        found for the given event and station combination.

        :type event_name: str
        :param event_name: The name of the event.
        :type station_id: str
        :param station_id: The id of the station in question.

        :rtype: dict
        :returns: Return a dictionary with "processed" and "synthetic" keys.
            Both values will be a list of strings. In the case of "processed"
            it will be a list of all available preprocessing tags. In the case
            of the synthetics it will be a list of all iterations for which
            synthetics are available.
        """
        if event_name not in self.events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)

        # Get some station information. This essentially assures the
        # availability of the raw data.
        try:
            self.get_stations_for_event(event_name, station_id)
        except:
            msg = "No raw data found for event '%s' and station '%s'." % (
                event_name, station_id)
            raise LASIFError(msg)

        # Get the waveform
        waveform_cache = self._get_waveform_cache_file(event_name, "raw")

        def get_components(waveform_cache):
            return [_i["channel"][-1] for _i in
                    waveform_cache.get_files_for_station(
                        *station_id.split("."))]

        raw_comps = sorted(get_components(waveform_cache), reverse=True)

        # Collect all tags and iteration names.
        all_files = {
            "raw": {"raw": raw_comps},
            "processed": {},
            "synthetic": {}}

        # Get the processed tags.
        data_dir = os.path.join(self.paths["data"], event_name)
        for tag in os.listdir(data_dir):
            # Only interested in preprocessed data.
            if not tag.startswith("preprocessed") or \
                    tag.endswith("_cache.sqlite"):
                continue
            waveforms = self._get_waveform_cache_file(event_name, tag)
            comps = get_components(waveforms)
            if comps:
                all_files["processed"][tag] = sorted(comps, reverse=True)

        # Get all synthetic files for the current iteration and event.
        synthetic_coordinates_mapping = {"X": "N", "Y": "E", "Z": "Z"}
        iterations = self.get_iteration_dict().keys()
        for iteration_name in iterations:
            synthetic_files = self._get_synthetic_waveform_filenames(
                event_name, iteration_name)
            if station_id not in synthetic_files:
                continue
            all_files["synthetic"][iteration_name] = \
                sorted([synthetic_coordinates_mapping[i]
                        for i in synthetic_files[station_id].keys()],
                       reverse=True)
        return all_files

    def plot_station(self, station_id, event_name):
        """
        Plots data for a single station and event.
        """
        from lasif.visualization import plot_data_for_station

        if event_name not in self.events:
            msg = "Event '%s' not found in project." % event_name
            raise ValueError(msg)

        # Get information about the station.
        station = self.get_stations_for_event(
            event_name, station_id=station_id)
        station["id"] = station_id

        available_data = self.discover_available_data(event_name, station_id)
        if not available_data:
            msg = ("No data available for the chosen event - station "
                   "combination")
            raise LASIFError(msg)

        # Callback for dynamic data plotting.
        def get_data_callback(data_type, tag_or_iteration=None):
            if data_type == "raw":
                return self.get_waveform_data(
                    event_name, station_id, data_type="raw")
            elif data_type == "processed":
                return self.get_waveform_data(
                    event_name, station_id, data_type="processed",
                    tag=tag_or_iteration)
            elif data_type == "synthetic":
                return self.get_waveform_data(event_name, station_id,
                                              data_type="synthetic",
                                              iteration_name=tag_or_iteration)

        # Plot it.
        plot_data_for_station(
            station=station,
            available_data=available_data,
            event=self.events[event_name],
            get_data_callback=get_data_callback,
            domain_bounds=self.domain)


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

        # =====================================================================
        # read iteration xml file, get event and list of stations
        # =====================================================================

        iteration = self._get_iteration(iteration_name)

        # Check that the event is part of the iterations.
        if event_name not in iteration.events:
            msg = ("Event '%s' not part of iteration '%s'.\nEvents available "
                   "in iteration:\n\t%s" %
                   (event_name, iteration_name, "\n\t".join(
                    sorted(iteration.events.keys()))))
            raise ValueError(msg)

        event = self.events[event_name]
        stations_for_event = iteration.events[event_name]["stations"].keys()

        # Get all stations and create a dictionary for the input file
        # generator.
        stations = self.get_stations_for_event(event_name)
        stations = [{
            "id": key, "latitude": value["latitude"],
            "longitude": value["longitude"],
            "elevation_in_m": value["elevation_in_m"],
            "local_depth_in_m": value["local_depth_in_m"]} for key, value in
            stations.iteritems() if key in stations_for_event]

        # =====================================================================
        # set solver options
        # =====================================================================

        solver = iteration.solver_settings

        # Currently only SES3D 4.1 is supported
        solver_format = solver["solver"].lower()
        if solver_format not in ["ses3d 4.1", "ses3d 2.0",
                                 "specfem3d cartesian"]:
            msg = ("Currently only SES3D 4.1, SES3D 2.0, and SPECFEM3D "
                   "CARTESIAN are supported.")
            raise ValueError(msg)
        solver_format = solver_format.replace(' ', '_')
        solver_format = solver_format.replace('.', '_')

        solver = solver["solver_settings"]

        # =====================================================================
        # create the input file generator, add event and stations,
        # populate the configuration items
        # =====================================================================

        # Add the event and the stations to the input file generator.
        gen = InputFileGenerator()
        gen.add_events(event["filename"])
        gen.add_stations(stations)

        if solver_format in ["ses3d_4_1", "ses3d_2_0"]:
            # event tag
            gen.config.event_tag = event_name

            # Time configuration.
            npts = solver["simulation_parameters"]["number_of_time_steps"]
            delta = solver["simulation_parameters"]["time_increment"]
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

            # Visco-elastic dissipation
            diss = solver["simulation_parameters"]["is_dissipative"]
            gen.config.is_dissipative = diss

            # Only SES3D 4.1 has the relaxation parameters.
            if solver_format == "ses3d_4_1":
                gen.config.Q_model_relaxation_times = \
                    solver["relaxation_parameter_list"]["tau"]
                gen.config.Q_model_weights_of_relaxation_mechanisms = \
                    solver["relaxation_parameter_list"]["w"]

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

            # Make source time function
            gen.config.source_time_function = \
                iteration.get_source_time_function()["data"]
        elif solver_format == "specfem3d_cartesian":
            gen.config.NSTEP = \
                solver["simulation_parameters"]["number_of_time_steps"]
            gen.config.DT = \
                solver["simulation_parameters"]["time_increment"]
            gen.config.NPROC = \
                solver["computational_setup"]["number_of_processors"]
            if simulation_type == "normal simulation":
                msg = ("'normal_simulation' not supported for SPECFEM3D "
                       "Cartesian. Please choose either 'adjoint_forward' or "
                       "'adjoint_reverse'.")
                raise NotImplementedError(msg)
            elif simulation_type == "adjoint forward":
                gen.config.SIMULATION_TYPE = 1
            elif simulation_type == "adjoint reverse":
                gen.config.SIMULATION_TYPE = 2
            else:
                raise NotImplementedError
            solver_format = solver_format.upper()
        else:
            msg = "Unknown solver."
            raise NotImplementedError(msg)

        # =================================================================
        # output
        # =================================================================
        output_dir = self.get_output_folder(
            "input_files___ITERATION_%s__%s__EVENT_%s" % (
                iteration_name, simulation_type.replace(" ", "_"),
                event_name))

        gen.write(format=solver_format, output_dir=output_dir)
        print "Written files to '%s'." % output_dir


    def _get_coordinates_for_waveform_file(self, waveform_filename,
                                           waveform_type, network_code,
                                           station_code, event_name):
        """
        Internal function used to grab station coordinates from the various
        sources.

        Used to make sure the same functionality is used everywhere.

        :param waveform_filename: The absolute filename of the waveform.
        :param waveform_type: The type of waveform file. Could be deduced from
            the pathname but probably not necessary as the information is
            likely readily available in situations where this method is called.
            One of "raw", "processed", and "synthetic"
        :param network_code: The network id of the file. Same reasoning as for
            the waveform_type.
        :param station_code: The station id of the file. Same reasoning as for
            the waveform_type.
        :param event_name: The name of the event. Same reasoning as for the
            waveform_type.

        Returns a dictionary containing "latitude", "longitude",
        "elevation_in_m", "local_depth_in_m". Returns None if no coordinates
        could be found.
        """
        if not os.path.exists(waveform_filename):
            msg = "Could not find the file '%s'" % waveform_filename
            raise ValueError(msg)

        # Attempt to first retrieve the coordinates from the station files.
        try:
            coordinates = self.station_cache.get_coordinates_for_station(
                network_code, station_code)
        except:
            pass
        else:
            return coordinates

        # The next step is to check for coordinates in sac files. For raw data
        # files, this is easy. For processed and synthetic data, the
        # corresponding raw data files will be attempted to be retrieved.
        cache = self._get_waveform_cache_file(event_name, "raw")

        if waveform_type in ("synthetic", "processed"):
            files = cache.get_files_for_station(network_code, station_code)
            if not files:
                waveform_cache_entry = None
            else:
                waveform_cache_entry = files[0]
        else:
            waveform_cache_entry = cache.get_details(waveform_filename)
            if waveform_cache_entry:
                waveform_cache_entry = waveform_cache_entry[0]

        if waveform_cache_entry and waveform_cache_entry["latitude"]:
            return {
                "latitude": waveform_cache_entry["latitude"],
                "longitude": waveform_cache_entry["longitude"],
                "elevation_in_m": waveform_cache_entry["elevation_in_m"],
                "local_depth_in_m": waveform_cache_entry["local_depth_in_m"]}

        # Last but not least resort to the Inventory Database.
        else:
            from lasif.tools.inventory_db import get_station_coordinates
            # Now check if the station_coordinates are available in the
            # inventory DB and use those.
            coords = get_station_coordinates(
                self.paths["inv_db_file"],
                ".".join((network_code, station_code)),
                self.paths["cache"],
                self.config["download_settings"]["arclink_username"])
            if coords:
                return coords
            else:
                return None

    def finalize_adjoint_sources(self, iteration_name, event_name):
        """
        Finalizes the adjoint sources.
        """

        from itertools import izip
        import numpy as np

        from lasif import rotations
        from lasif.window_manager import MisfitWindowManager
        from lasif.adjoint_src_manager import AdjointSourceManager

        # ====================================================================
        # initialisations
        # ====================================================================

        iteration = self._get_iteration(iteration_name)
        long_iteration_name = self._get_long_iteration_name(iteration_name)

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

        # loop through all the stations of this event
        for station_id, station in this_event["stations"].iteritems():

            try:
                this_station = all_stations[station_id]
            except KeyError:
                continue

            station_weight = station["station_weight"]
            windows = window_manager.get_windows_for_station(station_id)

            if not windows:
                msg = "No adjoint sources for station '%s'." % station_id
                warnings.warn(msg)
                continue

            all_channels = {}

            # loop through all channels for that station
            for channel_windows in windows:

                channel_id = channel_windows["channel_id"]
                cumulative_weight = 0
                all_data = []

                # loop through all windows of one channel
                for window in channel_windows["windows"]:

                    # get window properties
                    window_weight = window["weight"]
                    starttime = window["starttime"]
                    endtime = window["endtime"]
                    # load previously stored adjoint source
                    data = adj_src_manager.get_adjoint_src(channel_id,
                                                           starttime, endtime)
                    # lump all adjoint sources together
                    all_data.append(window_weight * data)
                    # compute cumulative weight of all windows for that channel
                    cumulative_weight += window_weight

                # apply weights for that channel
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

            # Rotate. if needed
            rec_lat = this_station["latitude"]
            rec_lng = this_station["longitude"]

            if self.domain["rotation_angle"]:
                # Rotate the adjoint source location.
                r_rec_lat, r_rec_lng = rotations.rotate_lat_lon(
                    rec_lat, rec_lng, self.domain["rotation_axis"],
                    -self.domain["rotation_angle"])
                # Rotate the adjoint sources.
                all_channels["N"], all_channels["E"], all_channels["Z"] = \
                    rotations.rotate_data(
                        all_channels["N"], all_channels["E"],
                        all_channels["Z"], rec_lat, rec_lng,
                        self.domain["rotation_axis"],
                        self.domain["rotation_angle"])
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
                for x, y, z in izip(-1.0 * all_channels[CHANNEL_MAPPING["X"]],
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
        """
        Return an iterator that returns processed data and synthetic files for
        one event and iteration.
        """
        from lasif.tools.data_synthetics_iterator import \
            DataSyntheticIterator
        return DataSyntheticIterator(self, event_name, iteration_name)

    def get_debug_information_for_file(self, filename):
        """
        Helper function returning a string with information LASIF knows about
        the file.

        Currently only works with waveform and station files.
        """
        from obspy import read, UTCDateTime
        import pprint

        err_msg = "LASIF cannot gather any information from the file."
        filename = os.path.abspath(filename)

        # Data file.
        if os.path.commonprefix([filename, self.paths["data"]]) == \
                self.paths["data"]:
            # Now split the path in event_name, tag, and filename. Any deeper
            # paths should not be allowed.
            rest, _ = os.path.split(os.path.relpath(filename,
                                                    self.paths["data"]))
            event_name, rest = os.path.split(rest)
            rest, tag = os.path.split(rest)
            # Now rest should not be existant anymore
            if rest:
                msg = "File is nested too deep in the data directory."
                raise LASIFError(msg)
            if tag.startswith("preprocessed_"):
                return ("The waveform file is a preprocessed file. LASIF will "
                        "not use it to extract any metainformation.")
            elif tag == "raw":
                # Get the corresponding waveform cache file.
                waveforms = self._get_waveform_cache_file(event_name, "raw",
                                                          use_cache=False)
                if not waveforms:
                    msg = "LASIF could not read the waveform file."
                    raise LASIFError(msg)
                details = waveforms.get_details(filename)
                if not details:
                    msg = "LASIF could not read the waveform file."
                    raise LASIFError(msg)
                filetype = read(filename)[0].stats._format
                # Now assemble the final return string.
                return (
                    "The {typ} file contains {c} channel{p}:\n"
                    "{channels}".format(
                        typ=filetype, c=len(details),
                        p="s" if len(details) != 1 else "",
                        channels="\n".join([
                            "\t{chan} | {start} - {end} | "
                            "Lat/Lng/Ele/Dep: {lat}/{lng}/"
                            "{ele}/{dep}".format(
                                chan=_i["channel_id"],
                                start=str(UTCDateTime(
                                    _i["starttime_timestamp"])),
                                end=str(UTCDateTime(_i["endtime_timestamp"])),
                                lat="%.2f" % _i["latitude"]
                                if _i["latitude"] is not None else "--",
                                lng="%.2f" % _i["longitude"]
                                if _i["longitude"] is not None else "--",
                                ele="%.2f" % _i["elevation_in_m"]
                                if _i["elevation_in_m"] is not None else "--",
                                dep="%.2f" % _i["local_depth_in_m"]
                                if _i["local_depth_in_m"]
                                is not None else "--",
                            ) for _i in details])))
            else:
                msg = "The waveform tag '%s' is not used by LASIF." % tag
                raise LASIFError(msg)

        # Station files.
        elif os.path.commonprefix([filename, self.paths["stations"]]) == \
                self.paths["stations"]:
            # Get the station cache
            details = self.station_cache.get_details(filename)
            if not details:
                raise LASIFError(err_msg)
            if filename in self.station_cache.files["resp"]:
                filetype = "RESP"
            elif filename in self.station_cache.files["seed"]:
                filetype = "SEED"
            elif filename in self.station_cache.files["stationxml"]:
                filetype = "STATIONXML"
            else:
                # This really should not happen.
                raise NotImplementedError
            # Now assemble the final return string.
            return (
                "The {typ} file contains information about {c} channel{p}:\n"
                "{channels}".format(
                    typ=filetype, c=len(details),
                    p="s" if len(details) != 1 else "",
                    channels="\n".join([
                        "\t{chan} | {start} - {end} | "
                        "Lat/Lng/Ele/Dep: {lat}/{lng}/"
                        "{ele}/{dep}".format(
                            chan=_i["channel_id"],
                            start=str(UTCDateTime(_i["start_date"])),
                            end=str(UTCDateTime(_i["end_date"]))
                            if _i["end_date"] else "--",
                            lat="%.2f" % _i["latitude"]
                            if _i["latitude"] is not None else "--",
                            lng="%.2f" % _i["longitude"]
                            if _i["longitude"] is not None else "--",
                            ele="%.2f" % _i["elevation_in_m"]
                            if _i["elevation_in_m"] is not None else "--",
                            dep="%.2f" % _i["local_depth_in_m"]
                            if _i["local_depth_in_m"] is not None else "--",
                        ) for _i in details])))

        # Event files.
        elif os.path.commonprefix([filename, self.paths["events"]]) == \
                self.paths["events"]:
            event_name = os.path.splitext(os.path.basename(filename))[0]
            try:
                event = self.events[event_name]
            except IndexError:
                raise LASIFError(err_msg)
            return (
                "The QuakeML files contains the following information:\n" +
                pprint.pformat(event)
            )

        else:
            raise LASIFError(err_msg)

    def has_station_file(self, channel_id, time):
        """
        Simple function returning True or False, if the channel specified with
        it's filename actually has a corresponding station file.
        """
        return self.station_cache.station_info_available(channel_id, time)

    def _get_all_raw_waveform_files_for_iteration(self, iteration_name):
        """
        Helper method returning a list of all raw waveform files for one
        iteration.
        """
        iteration = self._get_iteration(iteration_name)
        all_files = []
        for event_name, event in iteration.events.iteritems():
            waveforms = self._get_waveform_cache_file(event_name, "raw")
            if not waveforms:
                continue
            stations = event["stations"].keys()
            all_files.extend(
                [_i["filename"] for _i in waveforms.get_values()
                 if (_i["network"] + "." + _i["station"]) in stations])

        return all_files

    def _get_long_iteration_name(self, short_iteration_name):
        """
        Helper function for creating a long iteration name.

        Used for filenames and folder structure. Very simple and just used for
        consistencies sake.
        """
        return "ITERATION_%s" % short_iteration_name

    def _get_synthetic_waveform_filenames(self, event_name, iteration_name):
        """
        Helper function finding all stations for one simulation, e.g. one event
        and iteration combination.

        Currently only uses the filenames for distinction as the current SES3D
        version does not write that information in the file. Will have to be
        updated in due time as new solver are incorporated.
        """
        # First step is to get the folder.
        folder_name = os.path.join(self.paths["synthetics"], event_name,
                                   self._get_long_iteration_name(
                                       iteration_name))
        stations = {}
        for filename in glob.iglob(os.path.join(folder_name, "*")):
            try:
                network, station, _, component = [
                    _i.replace("_", "")
                    for _i in os.path.basename(filename).split(".")]
            except:
                msg = "File '%s' is not properly named. Will be skipped." % \
                    os.path.relpath(filename)
                warnings.warn(msg)
            station_id = "%s.%s" % (network, station)
            stations.setdefault(station_id, {})
            stations[station_id][component.upper()] = \
                os.path.abspath(filename)
        return stations

    def _calculate_adjoint_source(self, iteration_name, event_name,
                                  station_id, window_starttime,
                                  window_endtime):
        """
        Calculates the adjoint source for a certain window.
        """
        iteration = self._get_iteration(iteration_name)
        proc_tag = iteration.get_processing_tag()
        process_params = iteration.get_processing_tag()

        data = self.get_waveform_data(event_name, station_id,
                                      data_type="processed", tag=proc_tag)
        synthetics = self.get_waveform_data(event_name, station_id,
                                            data_type="synthetic",
                                            iteration_name=iteration_name)

        taper_percentage = 0.5
        data_trimmed = data.copy()\
            .trim(window_starttime, window_endtime)\
            .taper(type="cosine", max_percentage=0.5 * taper_percentage)
        synth_trim = synthetics.copy() \
            .trim(window_starttime, window_endtime) \
            .taper(type="cosine", max_percentage=0.5 * taper_percentage)


    def get_iteration_status(self, iteration_name):
        """
        Return a dictionary with information about the current status of an
        iteration.
        """
        iteration = self._get_iteration(iteration_name)
        proc_tag = iteration.get_processing_tag()

        # Dictionary collecting all the information.
        status = {}
        status["channels_not_yet_preprocessed"] = []
        status["stations_in_iteration_that_do_not_exist"] = []
        status["synthetic_data_missing"] = {}

        # Now check which events and stations are supposed to be part of the
        # iteration and try to find them and their corresponding preprocessed
        # counterparts.
        for event_name, event_info in iteration.events.iteritems():
            # Events with a weight of 0 are not considered.
            if event_info["event_weight"] == 0.0:
                continue
            # Get the existing files. This command can potentially be called
            # multiple times per instance with updates in-between. Thus caching
            # should be disabled.
            raw_waveforms = self._get_waveform_cache_file(event_name, "raw",
                                                          use_cache=False)
            proc_waveforms = self._get_waveform_cache_file(
                event_name, proc_tag, use_cache=False)
            # Extract the channels if some exist.
            raw_channels = {}
            if raw_waveforms:
                # Extract the channels.
                temp = [_i["channel_id"] for _i in raw_waveforms.get_values()]
                # Create a dictionary of all the channels sorted by station.
                for channel in temp:
                    station_id = ".".join(channel.split(".")[:2])
                    raw_channels.setdefault(station_id, [])
                    raw_channels[station_id].append(channel)
            # Extract the processed channels if some exist.
            if proc_waveforms:
                proc_channels = [_i["channel_id"] for _i in
                                 proc_waveforms.get_values()]
            else:
                proc_channels = []

            # Get the synthetics.
            synthetics = self._get_synthetic_waveform_filenames(event_name,
                                                                iteration_name)

            for station_id, station_info \
                    in event_info["stations"].iteritems():
                # Stations with a weight of zero are not considered.
                if station_info["station_weight"] == 0.0:
                    continue

                # Get all raw channels that have the current station.
                try:
                    current_chans = raw_channels[station_id]
                except KeyError:
                    current_chans = []

                # There should be at least one, otherwise the iteration xml
                # file is wrong.
                if not current_chans:
                    status["stations_in_iteration_that_do_not_exist"].append(
                        "Event '%s': '%s'" % (event_name, station_id))
                    continue
                for chan in current_chans:
                    if chan in proc_channels:
                        continue
                    status["channels_not_yet_preprocessed"].append(
                        "Event '%s': '%s'" % (event_name, chan))

                # Each station requires all three synthetic components. This is
                # necessary for rotations.
                if (station_id not in synthetics) or \
                        (len(synthetics[station_id]) != 3):
                    status["synthetic_data_missing"].setdefault(event_name, [])
                    status["synthetic_data_missing"][event_name].append(
                        station_id)
        return status
