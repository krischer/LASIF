#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION

:copyright: Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license: GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import glob
from itertools import izip
import os
import warnings


class EventPseudoDict(object):
    """
    A helper class for the main lasif.project.Project class.

    It aims to provide a convenient and simple interface to access all events
    within LASIF. It will work with all xml files in one folder.
    """
    def __init__(self, event_folder):
        """
        """
        # Build a dictionary with the keys being the names of all events and
        # the values the filenames.
        self.__event_files = {}
        for filename in glob.iglob(os.path.join(event_folder, "*.xml")):
            filename = os.path.abspath(filename)
            event_name = os.path.splitext(os.path.basename(filename))[0]
            self.__event_files[event_name] = filename

        # Cache the event information so everything is only read once at max.
        self.__event_info_cache = {}

    def keys(self):
        """
        Returns all event names.
        """
        return list(self.iterkeys())

    def iterkeys(self):
        """
        Like keys, but an iterator.
        """
        return (_i for _i in self.__event_files.iterkeys())

    def __contains__(self, event_name):
        if event_name in self.__event_files:
            return True
        return False

    def __len__(self):
        return len(self.__event_files)

    def values(self):
        """
        Returns a list with a dictionary describing each event.
        """
        return list(self.itervalues())

    def itervalues(self):
        """
        Like values(), but an iterator.
        """
        return (self.__getitem__(_i) for _i in self.__event_files.iterkeys())

    def items(self):
        """
        Returns a list of all items.
        """
        return list(self.iteritems())

    def iteritems(self):
        """
        Like items(), but an iterator.
        """
        return izip(self.iterkeys(), self.itervalues())

    def __getitem__(self, event_name):
        """
        Get information about one event.

        :type event_name: dict
        :param event_name: The event name.
        :rtype: dict
        :returns: A dictionary with information about the current event.
            Contains the following keys:
            * event_name
            * latitude
            * longitude
            * origin_time
            * depth_in_km
            * magnitude
            * region
            * magnitude_type
            * m_rr
            * m_tt
            * m_pp
            * m_rt
            * m_rp
            * m_tp

        The moment tensor components are in Newton * meter.
        """
        # Raise a KeyError if the event is not found.
        if event_name not in self.__event_files:
            raise KeyError("'%s'" % str(event_name))

        # Check if it already exists within the cache.
        if event_name in self.__event_info_cache:
            return self.__event_info_cache[event_name]

        # Import here so a simple class import does not yet require and ObsPy
        # import. Repeated imports are fast.
        from obspy import readEvents
        from obspy.core.util import FlinnEngdahl

        # Get an ObsPy event object.
        event = readEvents(self.__event_files[event_name])[0]

        # Extract information.
        mag = event.preferred_magnitude() or event.magnitudes[0]
        org = event.preferred_origin() or event.origins[0]
        if org.depth is None:
            warnings.warn("Origin contains no depth. Will be assumed to be 0")
            org.depth = 0.0
        if mag.magnitude_type is None:
            warnings.warn("Magnitude has no specified type. Will be assumed "
                          "to be Mw")
            mag.magnitude_type = "Mw"

        # Get the moment tensor.
        fm = event.preferred_focal_mechanism() or event.focal_mechanisms[0]
        mt = fm.moment_tensor.tensor

        info = {
            "event_name": event_name,
            "latitude": org.latitude,
            "longitude": org.longitude,
            "origin_time": org.time,
            "depth_in_km": org.depth / 1000.0,
            "magnitude": mag.mag,
            "region": FlinnEngdahl().get_region(org.longitude, org.latitude),
            "magnitude_type": mag.magnitude_type,
            "m_rr": mt.m_rr,
            "m_tt": mt.m_tt,
            "m_pp": mt.m_pp,
            "m_rt": mt.m_rt,
            "m_rp": mt.m_rp,
            "m_tp": mt.m_tp}

        # Store in cache and return.
        self.__event_info_cache[event_name] = info
        return info
