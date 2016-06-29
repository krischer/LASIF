#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os
import shutil

from .component import Component
from lasif import LASIFNotFoundError


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

    def get(self, iteration, event):
        iteration = self.comm.iterations.get(iteration)
        kernel_dir = os.path.join(self._folder, iteration.long_name, event)
        if not os.path.exists(kernel_dir) or not os.path.isdir(kernel_dir):
            raise LASIFNotFoundError("Kernel for iteration %s and event %s "
                                     "not found" % (iteration.long_name,
                                                    event))
        return kernel_dir

    def assert_has_boxfile(self, iteration, event):
        """
        Makes sure the kernel in question has a boxfile. Otherwise it will
        search all models until it finds one with the same dimensions and
        copy the boxfile.

        :param iteration: The iteration.
        :param event: The event name.
        """
        kernel_dir = self.get(iteration=iteration, event=event)
        boxfile = os.path.join(kernel_dir, "boxfile")
        if os.path.exists(boxfile):
            return

        boxfile_found = False

        # Find all vp gradients.
        vp_gradients = glob.glob(os.path.join(kernel_dir, "grad_csv_*"))
        file_count = len(vp_gradients)
        filesize = list(set([os.path.getsize(_i) for _i in vp_gradients]))
        if len(filesize) != 1:
            msg = ("The grad_cp_*_* files in '%s' do not all have "
                   "identical size.") % kernel_dir
            raise ValueError(msg)
        filesize = filesize[0]
        # Now loop over all model directories until a fitting one if found.
        for model_name in self.comm.models.list():
            model_dir = self.comm.models.get(model_name)
            # Use the lambda parameter files. One could also use any of the
            # others.
            lambda_files = glob.glob(os.path.join(model_dir, "lambda*"))
            if len(lambda_files) != file_count:
                continue
            l_filesize = list(
                set([os.path.getsize(_i) for _i in lambda_files]))
            if len(l_filesize) != 1 or l_filesize[0] != filesize:
                continue
            model_boxfile = os.path.join(model_dir, "boxfile")
            if not os.path.exists(model_boxfile):
                continue
            boxfile_found = True
            boxfile = model_boxfile
            break
        if boxfile_found is not True:
            msg = (
                "Could not find a suitable boxfile for the kernel stored "
                "in '%s'. Please either copy a suitable one to this "
                "directory or add a model with the same dimension to LASIF. "
                "LASIF will then be able to figure out the dimensions of it.")
            raise LASIFNotFoundError(msg)
        print("Copied boxfile from '%s' to '%s'." % (model_dir, kernel_dir))
        shutil.copyfile(boxfile, os.path.join(kernel_dir, "boxfile"))

    def get_model_handler(self, iteration, event):
        """
        Gets a an initialized SES3D model handler.

        :param iteration: The iteration.
        :param event: The event.
        """
        self.assert_has_boxfile(iteration=iteration, event=event)

        from lasif.ses3d_models import RawSES3DModelHandler  # NOQA

        return RawSES3DModelHandler(
            directory=self.get(iteration=iteration, event=event),
            domain=self.comm.project.domain,
            model_type="kernel")
