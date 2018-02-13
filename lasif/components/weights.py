#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os

from lasif import LASIFNotFoundError, LASIFError
from .component import Component
from obspy.geodetics import locations2degrees


class WeightsComponent(Component):
    """
    Component dealing with station and event weights.

    :param weights_folder: The folder with the weight toml files.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """

    def __init__(self, weights_folder, communicator, component_name):
        self.__cached_weights = {}
        self._folder = weights_folder
        super(WeightsComponent, self).__init__(communicator, component_name)

    def get_filename_for_weight_set(self, weight_set):
        """
        Helper function returning the filename of a weight set.
        """
        long_weight_set_name = self.get_long_weight_set_name(weight_set)
        folder = self._get_folder_for_weight_set(weight_set)
        return os.path.join(
            folder, long_weight_set_name + os.path.extsep + "toml")

    def _get_folder_for_weight_set(self, weight_set_name):
        """
        Helper function returning the path of a weights folder.
        """
        long_weight_set_name = self.get_long_weight_set_name(weight_set_name)
        folder = os.path.join(self.comm.project.paths["weights"],
                              long_weight_set_name)
        return folder

    def create_folder_for_weight_set(self, weight_set_name):
        long_weight_set_name = self.get_long_weight_set_name(weight_set_name)
        folder = os.path.join(self.comm.project.paths["weights"],
                              long_weight_set_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def get_weight_set_dict(self):
        """
        Returns a dictionary with the keys being the weight_set names and the
        values the weight_set filenames.
        """
        files = [os.path.abspath(_i) for _i in glob.iglob(os.path.join(
            self.comm.project.paths["weights"],
            "WEIGHTS_*/WEIGHTS_*%stoml" % os.extsep))]
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
        Get a list of all weight sets managed by this component.
        """
        return sorted(self.get_weight_set_dict().keys())

    def count(self):
        """
        Get the number of weight sets managed by this component.
        """
        return len(self.get_weight_set_dict())

    def has_weight_set(self, weight_set_name):
        """
        Test for existence of a weight_set.
        :type weight_set_name: str
        :param weight_set_name: The name of the weight_set.
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

        :param weight_set_name: The name of the weight set.
        :param events_dict: A dictionary specifying the used events.
        """
        weight_set_name = str(weight_set_name)
        if weight_set_name in self.get_weight_set_dict():
            msg = "Weight set %s already exists." % weight_set_name
            raise LASIFError(msg)

        self.create_folder_for_weight_set(weight_set_name)

        from lasif.weights_toml import create_weight_set_toml_string
        with open(self.get_filename_for_weight_set(weight_set_name),
                  "wt") as fh:
            fh.write(
                create_weight_set_toml_string(weight_set_name, events_dict))

    def change_weight_set(self, weight_set_name, weight_set, events_dict):
        """
        Changes an existing weight set. Writes into a tempfile and if
        successful it will replace the old file with the tempfile.

        :param weight_set_name: The name of the weight set.
        :param weight_set: The actual weight set
        :param events_dict: A dictionary specifying the used events.
        """
        weight_set_name = str(weight_set_name)

        from lasif.weights_toml import replace_weight_set_toml_string
        temp = self.get_filename_for_weight_set(weight_set_name)
        temp = temp + "_tmp"
        with open(temp, "w+") as fh:
            fh.write(
                replace_weight_set_toml_string(weight_set_name, events_dict,
                                               weight_set))
        os.remove(self.get_filename_for_weight_set(weight_set_name))
        os.rename(temp, self.get_filename_for_weight_set(weight_set_name))

    def calculate_station_weight(self, station, stations):
        """
        Calculates the weight set for a set of stations for one event
        :param station: A station ID
        :param stations: A dictionary with all the stations for this event
        :return: weight. weight for this specific station
        """

        factor = 0.0
        lat_1 = stations[station]["latitude"]
        lon_1 = stations[station]["longitude"]
        for stat in stations:
            lat_2 = stations[stat]["latitude"]
            lon_2 = stations[stat]["longitude"]
            if lat_1 == lat_2 and lon_1 == lon_2:
                continue
            distance = locations2degrees(lat_1, lon_1, lat_2, lon_2)
            factor += 1.0 / (1.0 + distance)
        weight = 1.0 / factor

        return weight

    def get(self, weight_set_name):
        """
        Returns a weight_set object.

        :param iteration_name: The name of the iteration to retrieve.
        """
        # Make it work with both the long and short version of the iteration
        # name, and existing iteration object.
        try:
            weight_set_name = str(weight_set_name.weight_set_name)
        except AttributeError:
            weight_set_name = str(weight_set_name)
            weight_set_name = weight_set_name.lstrip("WEIGHTS_")

        # Access cache.
        if weight_set_name in self.__cached_weights:
            return self.__cached_weights[weight_set_name]

        weights_dict = self.get_weight_set_dict()
        if weight_set_name not in weights_dict:
            msg = "Weights '%s' not found." % weight_set_name
            raise LASIFNotFoundError(msg)

        from lasif.weights_toml import WeightSet
        weight_set = WeightSet(weights_dict[weight_set_name])

        # Store in cache.
        self.__cached_weights[weight_set_name] = weight_set

        return weight_set
