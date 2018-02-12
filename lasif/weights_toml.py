#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functionality to deal with Weights toml files.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from collections import OrderedDict

import os
import re


class WeightSet(object):
    def __init__(self, weight_set_toml_filename):
        """
        Init function takes a Weights TOML file.
        """
        if not os.path.exists(weight_set_toml_filename):
            msg = "File '%s' not found." % weight_set_toml_filename
            raise ValueError(msg)
        self._parse_weights_toml(weight_set_toml_filename)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.__dict__ == other.__dict__:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _parse_weights_toml(self, weights_toml_filename):
        """
        Parses the given weights toml file and stores the information with the
        class instance.
        """
        import toml
        self.weight_info = toml.load(weights_toml_filename)

        # The weight_set name is dependent on the filename.
        self.weight_set_name = re.sub(r"\.toml$", "", re.sub(
            r"^WEIGHTS_", "", os.path.basename(weights_toml_filename)))

        self.description = self.weight_info['weight_set']['description']
        self.comments = self.weight_info['weight_set']['comment']

        self.events = OrderedDict()
        for event in self.weight_info['event']:
            event_name = event['name']
            self.events[event_name] = {
                "event_weight": float(event['weight']),
                "stations": OrderedDict()}
            if 'station' in event:
                for station in event['station']:
                    station_id = station['ID']
                    self.events[event_name]["stations"][station_id] = {
                        "station_weight": float(station['weight'])}

    @property
    def long_name(self):
        return "WEIGHTS%s" % self.name

    @property
    def name(self):
        return self.weight_set_name

    def __str__(self):
        """
        Pretty printing.
        """
        ret_str = (
            "LASIF weight_set\n"
            "\tName: {self.weight_set_name}\n"
            "\tDescription: {self.description}\n"
            "{comments}"
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
            event_count=len(self.events),
            pair_count=len(all_stations),
            station_count=len(set(all_stations)))


def create_weight_set_toml_string(weight_set_name, events_dict):
    toml_string = "# This is the weights set file.\n\n"
    weights_str = f"[weight_set]\n" \
                  f"  name = \"{weight_set_name}\"\n" \
                  f"  description = \"\"\n" \
                  f"  comment = \"\"\n\n"

    toml_string += weights_str
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


def replace_weight_set_toml_string(weight_set_name, events_dict, w_set):
    toml_string = "# This is the weights set file.\n\n"
    weights_str = f"[weight_set]\n" \
                  f"  name = \"{weight_set_name}\"\n" \
                  f"  description = \"\"\n" \
                  f"  comment = \"\"\n\n"

    toml_string += weights_str
    for event_name, stations in events_dict.items():
        event_string = f"[[event]]\n" \
                       f"  name = \"{event_name}\"\n" \
                       f"  weight = 1.0\n\n"
        for station in stations:
            we = \
                w_set.events[event_name]["stations"][station]["station_weight"]
            event_string += f"  [[event.station]]\n" \
                            f"    ID = \"{station}\"\n" \
                            f"    weight = {we} \n\n"
        toml_string += event_string

    return toml_string
