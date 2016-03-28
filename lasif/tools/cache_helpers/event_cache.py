#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Event cache. LASIF just feels much faster using one when there are more events.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import glob
import os
import warnings

import obspy

from lasif import LASIFError, LASIFWarning
from .file_info_cache import FileInfoCache


class EventCacheError(LASIFError):
    pass


class EventCache(FileInfoCache):
    """
    Cache for event files. Only supports

    Currently supports SEED, XML-SEED, RESP, and StationXML files.

    SEED files have to match the following pattern: 'dataless.*'
    RESP files: RESP.*
    StationXML: *.xml
    """
    def __init__(self, cache_db_file, event_folder, root_folder, read_only):
        self.index_values = [
            ("filename", "TEXT"),
            ("event_name", "TEXT"),
            ("latitude", "REAL"),
            ("longitude", "REAL"),
            ("depth_in_km", "REAL"),
            ("origin_time", "REAL"),
            ("m_rr", "REAL"),
            ("m_pp", "REAL"),
            ("m_tt", "REAL"),
            ("m_rp", "REAL"),
            ("m_rt", "REAL"),
            ("m_tp", "REAL"),
            ("magnitude", "REAL"),
            ("magnitude_type", "TEXT"),
            ("region", "TEXT")]

        self.filetypes = ["quakeml"]

        self.event_folder = event_folder

        super(EventCache, self).__init__(cache_db_file=cache_db_file,
                                         root_folder=root_folder,
                                         read_only=read_only,
                                         pretty_name="Event Cache",
                                         show_progress=False)

    def _find_files_quakeml(self):
        return glob.glob(os.path.join(self.event_folder, "*.xml"))

    @staticmethod
    def _extract_index_values_quakeml(filename):
        """
        Reads QuakeML files and extracts some keys per channel. Only one
        event per file is allows.
        """
        from obspy.geodetics import FlinnEngdahl

        try:
            cat = obspy.read_events(filename)
        except:
            msg = "Not a valid QuakeML file?"
            raise EventCacheError(msg)

        if len(cat) != 1:
            warnings.warn(
                "Each QuakeML file must have exactly one event. Event '%s' "
                "has %i. Only the first one will be parsed." % (
                    filename, len(cat)), LASIFWarning)

        event = cat[0]

        # Extract information.
        mag = event.preferred_magnitude() or event.magnitudes[0]
        org = event.preferred_origin() or event.origins[0]
        if org.depth is None:
            warnings.warn("Origin contains no depth. Will be assumed to be 0",
                          LASIFWarning)
            org.depth = 0.0
        if mag.magnitude_type is None:
            warnings.warn("Magnitude has no specified type. Will be assumed "
                          "to be Mw", LASIFWarning)
            mag.magnitude_type = "Mw"

        # Get the moment tensor.
        fm = event.preferred_focal_mechanism() or event.focal_mechanisms[0]
        mt = fm.moment_tensor.tensor

        event_name = os.path.splitext(os.path.basename(filename))[0]

        return [[
            str(filename),
            str(event_name),
            float(org.latitude),
            float(org.longitude),
            float(org.depth / 1000.0),
            float(org.time.timestamp),
            float(mt.m_rr),
            float(mt.m_pp),
            float(mt.m_tt),
            float(mt.m_rp),
            float(mt.m_rt),
            float(mt.m_tp),
            float(mag.mag),
            str(mag.magnitude_type),
            str(FlinnEngdahl().get_region(org.longitude, org.latitude))
        ]]
