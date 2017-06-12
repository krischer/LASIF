#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functionality to deal with Iteration XML files.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from collections import OrderedDict
from lxml import etree
from lxml.builder import E
import numpy as np
import os
import re

from lasif import LASIFError


class Iteration(object):

    def __init__(self, iteration_toml_filename, stf_fct):
        """
        Init function takes a Iteration XML file and the function to
        calculate the source time function..
        """
        if not os.path.exists(iteration_toml_filename):
            msg = "File '%s' not found." % iteration_toml_filename
            raise ValueError(msg)
        self._parse_iteration_toml(iteration_toml_filename)
        self.stf_fct = stf_fct

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.__dict__ == other.__dict__:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _parse_iteration_toml(self, iteration_toml_filename):
        """
        Parses the given iteration toml file and stores the information with the
        class instance.
        """
        import toml

        iter_dict = toml.load(iteration_toml_filename)

        print(iter_dict)


    def get_source_time_function(self):
        """
        Returns the source time function for the given iteration.

        Will return a dictionary with the following keys:
            * "delta": The time increment of the data.
            * "data": The actual source time function as an array.
        """
        delta = float(self.solver_settings["solver_settings"][
            "simulation_parameters"]["time_increment"])
        npts = int(self.solver_settings["solver_settings"][
            "simulation_parameters"]["number_of_time_steps"])

        freqmin = 1.0 / self.data_preprocessing["highpass_period"]
        freqmax = 1.0 / self.data_preprocessing["lowpass_period"]

        ret_dict = {"delta": delta}

        # Get source time function.
        ret_dict["data"] = self.stf_fct(
            npts=npts, delta=delta, freqmin=freqmin, freqmax=freqmax,
            iteration=self)
        # Some sanity checks as the function might be user supplied.
        if not isinstance(ret_dict["data"], np.ndarray):
            raise ValueError("Custom source time function does not return a "
                             "numpy array.")
        elif ret_dict["data"].dtype != np.float64:
            raise ValueError(
                "Custom source time function must have dtype `float64`. Yours "
                "has dtype `%s`." % (ret_dict["data"].dtype.__name__))
        elif len(ret_dict["data"]) != npts:
            raise ValueError(
                "Source time function must return a float64 numpy array with "
                "%i samples. Yours has %i samples." % (npts,
                                                       len(ret_dict["data"])))

        return ret_dict

    def get_process_params(self):
        """
        Small helper function retrieving the most important iteration
        parameters.
        """
        highpass = 1.0 / self.data_preprocessing["highpass_period"]
        lowpass = 1.0 / self.data_preprocessing["lowpass_period"]

        npts = self.solver_settings["solver_settings"][
            "simulation_parameters"]["number_of_time_steps"]
        dt = self.solver_settings["solver_settings"][
            "simulation_parameters"]["time_increment"]

        return {
            "highpass": float(highpass),
            "lowpass": float(lowpass),
            "npts": int(npts),
            "dt": float(dt)}

    @property
    def processing_tag(self):
        """
        Returns the processing tag for this iteration.
        """
        # Generate a preprocessing tag. This will identify the used
        # preprocessing so that duplicates can be avoided.
        processing_tag = ("preprocessed_hp_{highpass:.5f}_lp_{lowpass:.5f}_"
                          "npts_{npts}_dt_{dt:5f}")\
            .format(**self.get_process_params())
        return processing_tag

    @property
    def long_name(self):
        return "ITERATION_%s" % self.name

    @property
    def name(self):
        return self.iteration_name

    def __str__(self):
        """
        Pretty printing.
        """
        ret_str = (
            "LASIF Iteration\n"
            "\tName: {self.iteration_name}\n"
            "\tDescription: {self.description}\n"
            "{comments}"
            "\tPreprocessing Settings:\n"
            "\t\tHighpass Period: {hp:.3f} s\n"
            "\t\tLowpass Period: {lp:.3f} s\n"
            "\tSolver: {solver} | {timesteps} timesteps (dt: {dt}s)\n"
            "\t{event_count} events recorded at {station_count} "
            "unique stations\n"
            "\t{pair_count} event-station pairs (\"rays\")")

        comments = "\n".join("\tComment: %s" %
                             comment for comment in self.comments)
        if comments:
            comments += "\n"

        all_stations = []
        for ev in self.events.itervalues():
            all_stations.extend(ev["stations"].iterkeys())

        return ret_str.format(
            self=self, comments=comments,
            hp=self.data_preprocessing["highpass_period"],
            lp=self.data_preprocessing["lowpass_period"],
            solver=self.solver_settings["solver"],
            timesteps=self.solver_settings["solver_settings"][
                "simulation_parameters"]["number_of_time_steps"],
            dt=self.solver_settings["solver_settings"][
                "simulation_parameters"]["time_increment"],
            event_count=len(self.events),
            pair_count=len(all_stations),
            station_count=len(set(all_stations)))

