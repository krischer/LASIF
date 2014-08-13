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
        proc_tag = iteration.processing_tag
        process_params = iteration.get_process_params()

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
