#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import obspy
import os

from .component import Component
from lasif import LASIFNotFoundError
from lasif.tools.cache_helpers.event_cache import EventCache


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
        super(EventsComponent, self).__init__(communicator, component_name)
        self.__event_info_cache = {}
        self.folder = folder
        self.update_cache()

    def update_cache(self):
        """
        Clears the cached events. Events are only cached within one instance
        of the the EventsComponent in any case.
        """
        event_cache = EventCache(
            cache_db_file=os.path.join(
                self.comm.project.paths["cache"], "event_cache.sqlite"),
            root_folder=self.comm.project.paths["root"],
            read_only=self.comm.project.read_only_caches,
            event_folder=self.folder)

        values = event_cache.get_values()

        # Cache the event information so everything is only read once at max.
        self.__event_info_cache = {
            _i["event_name"]: _i for _i in values}
        for value in self.__event_info_cache.values():
            value["origin_time"] = obspy.UTCDateTime(value["origin_time"])

    def list(self):
        """
        List of all events.

        >>> comm = getfixture('events_comm')
        >>> comm.events.list() #  doctest: +NORMALIZE_WHITESPACE
        [u'GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11',
         u'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15']
        """
        return sorted(self.__event_info_cache.keys())

    def count(self):
        """
        Get the number of events managed by this component.

        >>> comm = getfixture('events_comm')
        >>> comm.events.count()
        2
        """
        return len(self.__event_info_cache)

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
        return event_name in self.__event_info_cache

    def get_all_events(self):
        """
        Returns a dictionary with the key being the event names and the
        values the information about each event, as would be returned by the
        :meth:`~lasif.components.events.EventsComponent.get` method.

        >>> comm = getfixture('events_comm')
        >>> comm.events.get_all_events() \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {u'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15': {...},
         u'GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11': {...}}
        """
        return copy.deepcopy(self.__event_info_cache)

    def get(self, event_name):
        """
        Get information about one event.

        This function uses multiple cache layers and is thus very cheap to
        call.

        :type event_name: str
        :param event_name: The name of the event.
        :rtype: dict

        >>> comm = getfixture('events_comm')
        >>> comm.events.get('GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15') \
        # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'m_tp': -2.17e+17, 'm_tt': 8.92e+17, 'depth_in_km': 7.0,
        'event_name': u'GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15',
        'region': u'TURKEY', 'longitude': 29.1,
        'filename': u'/.../GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15.xml',
        'm_rr': -8.07e+17, 'magnitude': 5.9, 'magnitude_type': u'Mwc',
        'latitude': 39.15,
        'origin_time': UTCDateTime(2011, 5, 19, 20, 15, 22, 900000),
        'm_rp': -5.3e+16, 'm_pp': -8.5e+16, 'm_rt': 2.8e+16}

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

        if event_name not in self.__event_info_cache:
            raise LASIFNotFoundError("Event '%s' not known to LASIF." %
                                     event_name)

        return self.__event_info_cache[event_name]
