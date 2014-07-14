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


# SES3D currently only identifies synthetics  via the filename. Use this
# template to get the name of a certain file.
SYNTHETIC_FILENAME_TEMPLATE = \
    "{network:_<2}.{station:_<5}.{location:_<3}.{component}"


class Project(object):
    """
    A class managing LASIF projects.

    It represents the heart of LASIF.
    """


    def finalize_adjoint_sources(self, iteration_name, event_name):
        """
        Finalizes the adjoint sources.
        """

        from itertools import izip
        import numpy as np

        from lasif import rotations
        from lasif.window_manager import WindowGroupManager
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
        window_manager = WindowGroupManager(window_directory,
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
