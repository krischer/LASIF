#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os

from lasif import LASIFNotFoundError, LASIFError
from .component import Component


class WeightsComponent(Component):
    """
    Component dealing with the iteration xml files.

    :param iterations_folder: The folder with the iteration XML files.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, weights_folder, communicator, component_name):
        self.__cached_iterations = {}
        self._folder = weights_folder
        super(WeightsComponent, self).__init__(communicator,
                                                  component_name)

    def get_filename_for_weight_set(self, weight_set):
        """
        Helper function returning the filename of a weight set.
        """
        long_weight_set_name = self.get_long_weight_set_name(weight_set)
        folder = self.get_folder_for_weight_set(weight_set)
        return os.path.join(
            folder, long_weight_set_name + os.path.extsep + "toml")

    def get_folder_for_weight_set(self, weight_set_name):
        """
        Helper function returning the path of an iteration folder.
        """
        long_weight_set_name = self.get_long_weight_set_name(weight_set_name)
        folder = os.path.join(self.comm.project.paths["weights"], long_weight_set_name)
        return folder

    def create_folder_for_weight_set(self, weight_set_name):
        long_weight_set_name = self.get_long_weight_set_name(weight_set_name)
        folder = os.path.join(self.comm.project.paths["weights"], long_weight_set_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def get_weight_set_dict(self):
        """
        Returns a dictionary with the keys being the iteration names and the
        values the iteration filenames.

        >>> import pprint
        >>> comm = getfixture('iterations_comm')
        >>> pprint.pprint(comm.iterations.get_iteration_dict()) \
        #  doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        {'1': '/.../ITERATION_1.xml',
         '2': '/.../ITERATION_2.xml'}
        """
        files = [os.path.abspath(_i) for _i in glob.iglob(os.path.join(
            self.comm.project.paths["weights"], "WEIGHTS_*/WEIGHTS_*%stoml" % os.extsep))]
        weight_dict = {os.path.splitext(os.path.basename(_i))[0][8:]: _i
                   for _i in files}
        return weight_dict

    def get_long_weight_set_name(self, weight_set_name):
        """
        Returns the long form of a weight set from its short name.
        """
        return "WEIGHTS_%s" % weight_set_name

    def list(self):
        """
        Get a list of all iterations managed by this component.

        >>> comm = getfixture('iterations_comm')
        >>> comm.weights.list()
        ['1', '2']
        """
        return sorted(self.get_weight_set_dict().keys())

    def count(self):
        """
        Get the number of weight sets managed by this component.

        >>> comm = getfixture('iterations_comm')
        >>> comm.weights.count()
        2
        """
        return len(self.get_weight_set_dict())

    def has_weight_set(self, weight_set_name):
        """
        Test for existence of a weight_set.

        :type weight_set_name: str
        :param weight_set_name: The name of the weight_set.

        >>> comm = getfixture('iterations_comm')
        >>> comm.weights.has_weight_set("A")
        True
        >>> comm.weights.has_iteration("99")
        False
        """
        # Make it work with both the long and short version of the iteration
        # name, and existing iteration object.
        try:
            weight_set_name = weight_set_name.weight_set_name
        except AttributeError:
            pass
        weight_set_name = weight_set_name.lstrip("WEIGHTS_")

        return weight_set_name in self.get_weight_set_dict()

    def create_new_weight_set(self, weight_set_name, events_dict):
        """
        Creates a new weight set.

        :param weight_set: The name of the weight set.
        :param events_dict: A dictionary specifying the used events.

        >>> comm = getfixture('iterations_comm')
        >>> comm.iterations.has_iteration("3")
        False
        >>> comm.iterations.create_new_iteration("3", "ses3d_4_1",
        ...     {"EVENT_1": ["AA.BB", "CC.DD"], "EVENT_2": ["EE.FF"]},
        ...     10.0, 20.0, quiet=True, create_folders=False)
        >>> comm.iterations.has_iteration("3")
        True
        >>> os.remove(comm.iterations.get_iteration_dict()["3"])
        """
        weight_set_name = str(weight_set_name)
        if weight_set_name in self.get_weight_set_dict():
            msg = "Weight set %s already exists." % weight_set_name
            raise LASIFError(msg)

        self.create_folder_for_weight_set(weight_set_name)

        from lasif.weights_toml import create_weight_set_toml_string
        with open(self.get_filename_for_weight_set(weight_set_name), "wt") as fh:
                fh.write(create_weight_set_toml_string(weight_set_name, events_dict))

    def save_iteration(self, iteration):
        """
        Save an iteration object to disc.
        :param iteration:
        """
        name = iteration.iteration_name
        filename = self.get_filename_for_iteration(name)
        iteration.write(filename)

        # Remove the iteration from the cache so it is loaded anew the next
        # time it is accessed.
        if name in self.__cached_iterations:
            del self.__cached_iterations[name]

    def get(self, weight_set_name):
        """
        Returns a weight_set object.

        :param iteration_name: The name of the iteration to retrieve.

        >>> comm = getfixture('iterations_comm')
        >>> comm.iterations.get("1")  # doctest: +ELLIPSIS
        <lasif.iteration_xml.Iteration object at ...>
        >>> print comm.iterations.get("1")  \
        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        LASIF Iteration
            Name: 1
            ...

        A :class:`~lasif.LASIFNotFoundError` will be raised, if the
        iteration is not known.

        >>> comm.iterations.get("99")
        Traceback (most recent call last):
            ...
        LASIFNotFoundError: ...


        It also works with the long iteration name and event an existing
        iteration object. This makes it simple to use, one path for all
        possibilities.
        >>> it = comm.iterations.get("ITERATION_1")
        >>> it  # doctest: +ELLIPSIS
        <lasif.iteration_xml.Iteration object at ...>
        >>> comm.iterations.get(it)
        <lasif.iteration_xml.Iteration object at ...>
        """
        # Make it work with both the long and short version of the iteration
        # name, and existing iteration object.
        try:
            weight_set_name = str(weight_set_name.weight_set_name)
        except AttributeError:
            weight_set_name = str(weight_set_name)
            weight_set_name = weight_set_name.lstrip("WEIGHTS_")

        # Access cache.
        if weight_set_name in self.__cached_iterations:
            return self.__cached_iterations[weight_set_name]

        weights_dict = self.get_weight_set_dict()
        if weight_set_name not in weights_dict:
            msg = "Weights '%s' not found." % weight_set_name
            raise LASIFNotFoundError(msg)

        from lasif.weights_toml import Iteration
        it = Iteration(weights_dict[weight_set_name],
                       stf_fct=self.comm.project.get_project_function(
                           "source_time_function"))

        # Store in cache.
        self.__cached_iterations[weight_set_name] = it

        return it
