#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os
import warnings

from .component import Component
from lasif import LASIFNotFoundError, LASIFWarning


class EventsComponent(Component):
    """
    Component managing a folder of QuakeML files.

    Each file must adhere to the scheme ``*.xml``.

    :param folder: Folder with QuakeML files.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the
        communicator.
    """
    def __init__(self, folder, communicator, component_name):
        # Build a dictionary with the keys being the names of all events and
        # the values the filenames.
        self.__event_files = {}
        for filename in glob.iglob(os.path.join(folder, "*.xml")):
            filename = os.path.abspath(filename)
            event_name = os.path.splitext(os.path.basename(filename))[0]
            self.__event_files[event_name] = filename

        # Cache the event information so everything is only read once at max.
        self.__event_info_cache = {}

        super(EventsComponent, self).__init__(communicator, component_name)

    def list(self):
        """
        List of all events.

        >>> comm = getfixture('events_comm')
        >>> comm.events.list() #  doctest: +NORMALIZE_WHITESPACE
        ['GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11',
         'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15']
        """
        return sorted(self.__event_files.keys())

    def count(self):
        """
        Get the number of events managed by this component.

        >>> comm = getfixture('events_comm')
        >>> comm.events.count()
        2
        """
        return len(self.__event_files)

    def has_event(self, event_name):
        """
        Test for existence of an event.

        :type event_name: str
        :param event_name: The name of the event.

        >>> comm = getfixture('events_comm')
        >>> comm.events.has_event('GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15')
        True
        >>> comm.events.has_event('random')
        False
        """
        # Make sure  it also works with existing event dictionaries. This
        # has the potential to simplify lots of code.
        try:
            event_name = event_name["event_name"]
        except (KeyError, TypeError):
            pass
        return event_name in self.__event_files

    def get_all_events(self):
        """
        Returns a dictionary with the key being the event names and the
        values the information about each event, as would be returned by the
        :meth:`~lasif.components.events.EventsComponent.get` method.

        >>> comm = getfixture('events_comm')
        >>> comm.events.get_all_events() \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15': {...},
         'GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11': {...}}
        """
        all_events = {}
        for event_name in self.__event_files.keys():
            all_events[event_name] = self.get(event_name)
        return all_events

    def get(self, event_name):
        """
        Get information about one event.

        This function caches the results and is thus very cheap.

        :type event_name: str
        :param event_name: The name of the event.
        :rtype: dict

        >>> comm = getfixture('events_comm')
        >>> comm.events.get('GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15') \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'m_tp': -2.17e+17, 'm_tt': 8.92e+17, 'depth_in_km': 7.0,
        'event_name': 'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15',
        'magnitude_type': 'Mwc', 'm_rr': -8.07e+17, 'm_rp': -5.3e+16,
        'm_pp': -8.5e+16, 'm_rt': 2.8e+16, 'region': u'TURKEY',
        'longitude': 29.1,
        'filename': '/.../GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15.xml',
        'magnitude': 5.9, 'latitude': 39.15,
        'origin_time': UTCDateTime(2011, 5, 19, 20, 15, 22, 900000)}

        The moment tensor components are in ``Nm``. The dictionary will
        contain the following keys:

        >>> sorted(comm.events.get(
        ...     'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15').keys()) \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
         ['depth_in_km', 'event_name', 'filename', 'latitude', 'longitude',
          'm_pp', 'm_rp', 'm_rr', 'm_rt', 'm_tp', 'm_tt', 'magnitude',
          'magnitude_type', 'origin_time', 'region']

        It also works with an existing event dictionary. This eases calling
        the function under certain circumstances.

        >>> ev = comm.events.get('GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15')
        >>> ev == comm.events.get(ev)
        True
        """
        # Make sure  it also works with existing event dictionaries. This
        # has the potential to simplify lots of code.
        try:
            event_name = event_name["event_name"]
        except (KeyError, TypeError):
            pass

        if event_name not in self.__event_files:
            raise LASIFNotFoundError("Event '%s' not known to LASIF." %
                                     event_name)

        # Check if it already exists within the cache.
        if event_name in self.__event_info_cache:
            return self.__event_info_cache[event_name]

        # Import here so a simple class import does not yet require and ObsPy
        # import. Repeated imports are fast as they do nothing.
        from obspy import readEvents
        from obspy.core.util import FlinnEngdahl

        # Get an ObsPy event object.
        filename = self.__event_files[event_name]
        cat = readEvents(filename)
        if len(cat) > 1:
            msg = "File '%s' has more than one event. Only the first one " \
                  "will be used"
            warnings.warn(msg % event_name, LASIFWarning)
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

        info = {
            "event_name": event_name,
            "filename": filename,
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
