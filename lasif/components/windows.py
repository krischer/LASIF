#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .component import Component


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
