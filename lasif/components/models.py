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

    def get_model_handler(self, model_name):
        """
        Gets a an initialized SES3D model handler.

        :param model_name: The name of the model.
        """
        from lasif.ses3d_models import RawSES3DModelHandler  # NOQA
        return RawSES3DModelHandler(
            directory=self.get(model_name),
            domain=self.comm.project.domain,
            model_type="earth_model")
