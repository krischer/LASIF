#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os

from lasif import LASIFNotFoundError, LASIFError
from .component import Component


class IterationsComponent(Component):
    """
    Component dealing with the iteration xml files.

    :param iterations_folder: The folder with the iteration XML files.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, communicator, component_name):
        self.__cached_iterations = {}
        super(IterationsComponent, self).__init__(communicator,
                                                  component_name)

    def get_long_iteration_name(self, iteration_name):
        """
        Returns the long form of an iteration from its short name.

        >>> comm = getfixture('iterations_comm')
        >>> comm.iterations.get_long_iteration_name("1")
        'ITERATION_1'
        """
        return "ITERATION_%s" % iteration_name

    def setup_directories_for_iteration(self, iteration_name):
        """
        Sets up the directory structure required for the iteration
        :param iteration_name: The iteration for which to create the folders.
        """
        long_iter_name = self.get_long_iteration_name(iteration_name)
        self.create_synthetics_folder_for_iteration(long_iter_name)
        self.create_input_files_folder_for_iteration(long_iter_name)
        self.create_adjoint_sources_and_windows_folder_for_iteration(long_iter_name)

    def create_synthetics_folder_for_iteration(self, long_iteration_name):
        """
        Create the synthetics folder if it does not yet exists.

        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["synthetics"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def create_input_files_folder_for_iteration(self, long_iteration_name):
        """
        Create the synthetics folder if it does not yet exists.

        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["input_files"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def create_adjoint_sources_and_windows_folder_for_iteration(self, long_iteration_name):
        """
        Create the adjoint_sources_and_windows folder if it does not yet exists.

        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["adjoint_sources"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)\
