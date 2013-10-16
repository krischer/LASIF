#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cache taking care of a single waveform directory.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import glob
from itertools import izip
import obspy
import os
import warnings

from lasif.tools.file_info_cache import FileInfoCache


class WaveformCache(FileInfoCache):
    """
    Cache taking care of a single waveform directory.

    Supports all waveform files readable with ObsPy.
    """
    def __init__(self, cache_db_file, waveform_folder, show_progress=True):
        self.index_values = [
            ("network", "TEXT"),
            ("station", "TEXT"),
            ("location", "TEXT"),
            ("channel", "TEXT"),
            ("channel_id", "TEXT"),
            ("starttime_timestamp", "REAL"),
            ("endtime_timestamp", "REAL"),
            ("latitude", "REAL"),
            ("longitude", "REAL"),
            ("elevation_in_m", "REAL"),
            ("local_depth_in_m", "REAL")]

        self.filetypes = ["waveform"]

        self.waveform_folder = waveform_folder

        super(WaveformCache, self).__init__(cache_db_file=cache_db_file,
                                            show_progress=show_progress)

    def get_files_for_station(self, network, station):
        """
        Returns a list of all files belonging to one station. If no station is
        found it will return an empty list.

        :type network: str
        :param network: The network id.
        :type station: str
        :param station: The station id.
        """
        query = """
        SELECT %s, filename
        FROM files
        INNER JOIN indices
        ON files.id=indices.filepath_id
        WHERE indices.network=? AND indices.station=?
        """ % ", ".join(["indices.%s" % _i[0] for _i in self.index_values])

        all_values = []
        indices = [_i[0] for _i in self.index_values]

        for _i in self.db_cursor.execute(query, (network, station)):
            values = {key: value for (key, value) in izip(indices, _i)}
            values["filename"] = _i[-1]
            all_values.append(values)

        return all_values

    def _find_files_waveform(self):
        return glob.glob(os.path.join(self.waveform_folder, "*"))

    def _extract_index_values_waveform(self, filename):
        """
        Extract all the information from the file.
        """
        try:
            st = obspy.read(filename)
        except:
            warnings.warn("Could not read waveform file '%s'." % filename)
            return None

        waveforms = []
        for tr in st:
            latitude = None
            longitude = None
            elevation_in_m = None
            local_depth_in_m = None

            # Use SAC coordinates if available.
            if "sac" in tr.stats:
                s_stats = tr.stats.sac
                # All or nothing with the exception of the depth.
                if -12345.0 not in [s_stats.stla, s_stats.stlo, s_stats.stel]:
                    latitude = float(s_stats.stla)
                    longitude = float(s_stats.stlo)
                    elevation_in_m = float(s_stats.stel)
                    local_depth_in_m = float(s_stats.stdp) \
                        if s_stats.stdp != -12345.0 else 0.0

            s = tr.stats

            # Special case handling if the location is set to "--".
            if s.location == "--":
                s.location = ""

            waveforms.append([
                s.network, s.station, s.location, s.channel, tr.id,
                s.starttime.timestamp, s.endtime.timestamp, latitude,
                longitude, elevation_in_m, local_depth_in_m])

        return waveforms
