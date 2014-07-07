#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

from .component import Component
from lasif import LASIFNotFoundError


class ModelsComponent(Component):
    """
    Component dealing with earth models.

    Needs access to the project component to get the rotation parameters.

    :param models_folder: The folder with the 3D Models.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, models_folder, communicator, component_name):
        self._folder = models_folder
        super(ModelsComponent, self).__init__(communicator,
                                              component_name)

    def list(self):
        """
        Get a list of all models managed by this component.
        """
        contents = os.listdir(self._folder)
        contents = [_i for _i in contents if os.path.isdir(os.path.join(
            self._folder, _i))]
        return sorted(contents)

    def get(self, model_name):
        model_dir = os.path.join(self._folder, model_name)
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            raise LASIFNotFoundError("Model '%s' not known." % model_name)
        return model_dir

    def plot(self, model_name):
        from lasif import ses3d_models

        model_dir = self.get(model_name)

        handler = ses3d_models.RawSES3DModelHandler(model_dir)
        domain = self.comm.project.domain
        handler.rotation_axis = domain["rotation_axis"]
        handler.rotation_angle_in_degree = domain["rotation_angle"]

        while True:
            print handler
            print ""

            inp = raw_input("Enter 'COMPONENT DEPTH' "
                            "('quit/exit' to exit): ").strip()
            if inp.lower() in ["quit", "q", "exit", "leave"]:
                break
            try:
                component, depth = inp.split()
            except:
                continue

            try:
                handler.parse_component(component)
            except:
                continue
            handler.plot_depth_slice(component, float(depth))
