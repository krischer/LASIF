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
from lxml import etree
from lxml.builder import E
import os

from lasif.project import LASIFException


class Iteration(object):
    def __init__(self, iteration_xml_filename):
        """
        Init function takes a Iteration XML file.
        """
        if not os.path.exists(iteration_xml_filename):
            msg = "File '%s' not found." % iteration_xml_filename
            raise ValueError(msg)
        self._parse_iteration_xml(iteration_xml_filename)

    def _parse_iteration_xml(self, iteration_xml_filename):
        """
        Parses the given iteration xml file and stores the information with the
        class instance.
        """
        root = etree.parse(iteration_xml_filename).getroot()

        self.iteration_name = self._get(root, "iteration_name")
        self.description = self._get_if_available(root,
            "iteration_description")
        self.comments = [_i.text for _i in root.findall("comment") if _i.text]

        self.data_preprocessing = {}
        prep = root.find("data_preprocessing")
        self.data_preprocessing["highpass_frequency"] = \
            float(self._get(prep, "highpass_frequency"))
        self.data_preprocessing["lowpass_frequency"] = \
            float(self._get(prep, "lowpass_frequency"))

    def _get(self, element, node_name):
        return element.find(node_name).text

    def _get_if_available(self, element, node_name):
        item = element.find(node_name)
        if item:
            return item.text
        return None


def create_iteration_xml_string(iteration_name, solver_name, events):
    """
    Creates a new iteration string.

    Returns a string containing the XML representation.

    :param iteration_name: The iteration name. Should start with a number.
    :param solver_name: The name of the solver to be used for this iteration.
    :param events: A dictionary. The key is the event name, and the value a
        list of station ids for this event.
    """
    solver_doc = _get_default_solver_settings(solver_name)
    if solver_name.lower() == "ses3d_4_0":
        solver_name = "SES3D 4.0"
    else:
        raise NotImplementedError

    # Loop over all events.
    events_doc = []
    # Also over all stations.
    for event_name, stations in events.iteritems():
        stations_doc = [E.station(
            E.station_id(station),
            E.station_weight("1.0"),
            E.time_correction_in_s("0.0")) for station in stations]
        events_doc.append(E.event(
            E.event_name(event_name),
            E.event_weight("1.0"),
            E.time_correction_in_s("0.0"),
            *stations_doc))

    doc = E.iteration(
        E.iteration_name(iteration_name),
        E.iteration_description(""),
        E.comment(""),
        E.data_preprocessing(
            E.highpass_frequency("0.01"),
            E.lowpass_frequency("0.5")),
        E.rejection_criteria(
            E.minimum_trace_length_in_s("500.0"),
            E.signal_to_noise(
                E.test_interval_from_origin_in_s("100.0"),
                E.max_amplitude_ratio("100.0"))
            ),
        E.source_time_function("Filtered Heaviside"),
        E.solver_parameters(
            E.solver(solver_name),
            solver_doc),
        *events_doc)

    string_doc = etree.tostring(doc, pretty_print=True,
        xml_declaration=True, encoding="UTF-8")
    return string_doc


def _get_default_solver_settings(solver):
    """
    Helper function returning etree representation of a solver's default
    settings.
    """
    known_solvers = ["ses3d_4_0A"]
    if solver.lower() == "ses3d_4_0":
        from lasif.utils import generate_ses3d_4_0_template
        return generate_ses3d_4_0_template()
    else:
        msg = "Solver '%s' not known. Known solvers: %s" % (solver,
            ",".join(known_solvers))
        raise LASIFException(msg)
