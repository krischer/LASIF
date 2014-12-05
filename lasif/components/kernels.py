#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

from .component import Component


class KernelsComponent(Component):
    """
    Component dealing with the kernels.

    :param models_folder: The folder with the 3D Models.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, kernels_folder, communicator, component_name):
        self._folder = kernels_folder
        super(KernelsComponent, self).__init__(communicator,
                                               component_name)

    def list(self):
        """
        Get a list of all kernels managed by this component.
        """
        kernels = []
        # Get the iterations first.
        contents = os.listdir(self._folder)
        contents = [_i for _i in contents if os.path.isdir(os.path.join(
            self._folder, _i))]
        contents = [_i for _i in contents if _i.startswith("ITERATION_")]
        contents = sorted(contents)
        for iteration in contents:
            events = os.listdir(os.path.join(self._folder, iteration))
            events = [_i for _i in events if os.path.isdir(os.path.join(
                self._folder, iteration, _i))]
            events = sorted(events)
            for event in events:
                kernels.append({"iteration": iteration.strip("ITERATION_"),
                                "event": event})
        return kernels
