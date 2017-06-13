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
        self.iteration_info = toml.load(iteration_toml_filename)

        # The iteration name is dependent on the filename.
        self.iteration_name = re.sub(r"\.toml$", "", re.sub(
            r"^ITERATION_", "", os.path.basename(iteration_toml_filename)))

        self.description = self.iteration_info['iteration']['description']
        self.comments = self.iteration_info['iteration']['comment']

        self.scale_data_to_synthetics = self.iteration_info['iteration']['scale_data_to_synthetics']

        # Defaults to True.
        if self.scale_data_to_synthetics is None:
            self.scale_data_to_synthetics = True
        if self.scale_data_to_synthetics is not True or False:
            raise ValueError("Value '%s' invalid for "
                             "'scale_data_to_synthetics'." %
                             self.scale_data_to_synthetics)

        self.data_preprocessing = self.iteration_info['data_preprocessing']

        self.events = OrderedDict()
        for event in self.iteration_info['event']:
            event_name = event['name']
            self.events[event_name] = {
                "event_weight": float(event['weight']),
                "stations": OrderedDict()}
            if 'station' in event:
                for station in event['station']:
                    station_id = station['ID']
                    self.events[event_name]["stations"][station_id] = {
                        "station_weight": float(station['weight'])}

    def get_process_params(self):
        """
        Small helper function retrieving the most important iteration
        parameters.
        """
        highpass = 1.0 / self.data_preprocessing["highpass_period"]
        lowpass = 1.0 / self.data_preprocessing["lowpass_period"]

        return {
            "highpass": float(highpass),
            "lowpass": float(lowpass)}

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
            "\t{event_count} events recorded at {station_count} "
            "unique stations\n"
            "\t{pair_count} event-station pairs (\"rays\")")

        comments = "\n".join("\tComment: %s" %
                             comment for comment in self.comments)
        if comments:
            comments += "\n"

        all_stations = []
        for ev in self.events.values():
            all_stations.extend(ev["stations"].keys())

        return ret_str.format(
            self=self, comments=comments,
            hp=self.data_preprocessing["highpass_period"],
            lp=self.data_preprocessing["lowpass_period"],
            event_count=len(self.events),
            pair_count=len(all_stations),
            station_count=len(set(all_stations)))

    def write(self, filename):
        """
        Serialized the Iteration structure once again.

        :param filename: The path that will be written to.
        """

        toml_string = "# This is the iteration file.\n\n"

        if self.scale_data_to_synthetics:
            scale_data_to_synthetics = "true"
        else:
            scale_data_to_synthetics = "false"

        iteration_str = f"[iteration]\n" \
                        f"  name = {self.iteration_name}\n" \
                        f"  description = \"\"\n" \
                        f"  comment = \"\"\n" \
                        f"  scale_data_to_synthetics = {scale_data_to_synthetics}\n\n"

        data_preproc_str = f"[data_preprocessing]\n" \
                           f"  highpass_period = {self.data_preprocessing['highpass_period']}\n" \
                           f"  lowpass_period = {self.data_preprocessing['lowpass_period']}\n" \
                           f"\n"


        toml_string += iteration_str + data_preproc_str
        for event, event_info in self.events.items():
            event_string = f"[[event]]\n" \
                           f"  name = \"{event}\"\n" \
                           f"  weight = {event_info['event_weight']}\n\n"

            for station, station_info in event_info['stations'].items():
                event_string += f"  [[event.station]]\n" \
                                f"    ID = \"{station}\"\n" \
                                f"    weight = {station_info['station_weight']} \n\n"
            toml_string += event_string

        with open(filename, "wt") as fh:
            fh.write(toml_string)

def create_iteration_toml_string(iteration_number, events_dict, min_period, max_period):
    toml_string = "# This is the iteration file.\n\n"
    iteration_str = f"[iteration]\n" \
                    f"  name = {iteration_number}\n" \
                    f"  description = \"\"\n" \
                    f"  comment = \"\"\n" \
                    f"  scale_data_to_synthetics = true\n\n"

    data_preproc_str = f"[data_preprocessing]\n" \
                       f"  highpass_period = {max_period}\n" \
                       f"  lowpass_period = {min_period}\n" \
                       f"\n"

    toml_string += iteration_str + data_preproc_str
    for event_name, stations in events_dict.items():
        event_string = f"[[event]]\n" \
                       f"  name = \"{event_name}\"\n" \
                       f"  weight = 1.0\n\n"
        for station in stations:
            event_string += f"  [[event.station]]\n" \
                            f"    ID = \"{station}\"\n" \
                            f"    weight = 1.0 \n\n"
        toml_string += event_string

    return toml_string

