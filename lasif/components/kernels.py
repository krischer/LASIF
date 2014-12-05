#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import os

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
        event = self.comm.events.get(event)
        kernel_dir = os.path.join(self._folder, iteration.long_name,
                                  event["event_name"])
        if not os.path.exists(kernel_dir) or not os.path.isdir(kernel_dir):
            raise LASIFNotFoundError("Kernel for iteration %s and event %s "
                                     "not found" % (iteration.long_name,
                                                    event["event_name"]))
        return kernel_dir

    def plot(self, iteration, event):
        from lasif import ses3d_models

        model_dir = self.get(iteration, event)

        handler = ses3d_models.RawSES3DModelHandler(
            model_dir, domain=self.comm.project.domain,
            model_type="kernel")

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
