#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os

from .component import Component
import shutil


class IterationsComponent(Component):
    """
    Component dealing with the iteration xml files.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, communicator, component_name):
        self.__cached_iterations = {}
        super(IterationsComponent, self).__init__(communicator,
                                                  component_name)

    def get_long_iteration_name(self, iteration_name):
        """
        Returns the long form of an iteration from its short or long name.

        >>> comm = getfixture('iterations_comm')
        >>> comm.iterations.get_long_iteration_name("1")
        'ITERATION_1'
        """

        iteration_name = iteration_name.lstrip("ITERATION_")
        return "ITERATION_%s" % iteration_name

    def setup_directories_for_iteration(self, iteration_name,
                                        remove_dirs=False):
        """
        Sets up the directory structure required for the iteration
        :param iteration_name: The iteration for which to create the folders.
        :param remove_dirs: Boolean if set to True the iteration is removed
        """
        long_iter_name = self.get_long_iteration_name(iteration_name)
        self._create_synthetics_folder_for_iteration(long_iter_name,
                                                     remove_dirs)
        self._create_input_files_folder_for_iteration(long_iter_name,
                                                      remove_dirs)
        self._create_adjoint_sources_and_windows_folder_for_iteration(
            long_iter_name, remove_dirs)

    def _create_synthetics_folder_for_iteration(self, long_iteration_name,
                                                remove_dirs=False):
        """
        Create the synthetics folder if it does not yet exist.
        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["eq_synthetics"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if remove_dirs:
            shutil.rmtree(folder)

    def _create_input_files_folder_for_iteration(self, long_iteration_name,
                                                 remove_dirs=False):
        """
        Create the synthetics folder if it does not yet exist.
        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["salvus_input"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if remove_dirs:
            shutil.rmtree(folder)

    def _create_adjoint_sources_and_windows_folder_for_iteration(
            self, long_iteration_name, remove_dirs=False):
        """
        Create the adjoint_sources_and_windows folder if it does not yet exist.
        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["adjoint_sources"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if remove_dirs:
            shutil.rmtree(folder)

    def list(self):
        """
        Returns a list of all the iterations known to LASIF.
        """
        files = [os.path.abspath(_i) for _i in glob.iglob(os.path.join(
            self.comm.project.paths["eq_synthetics"], "ITERATION_*"))]
        iterations = [os.path.splitext(os.path.basename(_i))[0][10:]
                      for _i in files]
        return sorted(iterations)

    def has_iteration(self, iteration_name):
        """
        Checks for existance of an iteration
        """
        iteration_name = iteration_name.lstrip("ITERATION_")
        if iteration_name in self.list():
            return True
        return False
