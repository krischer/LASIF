#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

from lasif import LASIFNotFoundError
from .component import Component
from ..window_manager import WindowGroupManager


class WindowsComponent(Component):
    """
    Component dealing with the windows.

    :param windows_folder: The folder where the windows are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, windows_folder, communicator, component_name):
        self._folder = windows_folder
        super(WindowsComponent, self).__init__(communicator,
                                               component_name)

    def list(self):
        """
        Lists all events with windows.
        """
        return sorted([_i for _i in os.listdir(self._folder) if
                       os.path.isdir(os.path.join(self._folder, _i))])

    def list_for_event(self, event_name):
        """
        List of all iterations with windows for an event.

        :param event_name: The name of the event.
        """
        event_folder = os.path.join(self._folder, event_name)
        if not os.path.exists(event_folder):
            msg = "No windows for event '%s'." % event_name
            raise LASIFNotFoundError(msg)

        return sorted([_i.lstrip("ITERATION_")
                       for _i in os.listdir(event_folder) if
                       os.path.isdir(os.path.join(event_folder, _i)) and
                       _i.startswith("ITERATION_")])

    def get(self, event_name, iteration_name):
        """
        Returns the window manager instance for a given event and iteration.

        :param event_name: The name of the event.
        :param iteration_name: The name of the iteration.
        """
        if not self.comm.events.has_event(event_name):
            msg = "Event '%s' not known." % event_name
            raise LASIFNotFoundError(msg)

        if not self.comm.iterations.has_iteration(iteration_name):
            msg = "Iteration '%s' not known." % iteration_name
            raise LASIFNotFoundError(msg)

        iteration = self.comm.iterations.get(iteration_name)
        if event_name not in iteration.events:
            msg = "Event '%s' not part of iteration '%s'." % (event_name,
                                                              iteration_name)
            raise ValueError(msg)

        folder = os.path.join(self._folder, event_name,
                              self.comm.iterations.get_long_iteration_name(
                                  iteration_name))
        return WindowGroupManager(folder, iteration_name, event_name)
