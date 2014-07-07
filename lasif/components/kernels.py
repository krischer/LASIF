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
        contents = os.listdir(self._folder)
        contents = [_i for _i in contents if os.path.isdir(os.path.join(
            self._folder, _i))]
        return sorted(contents)
