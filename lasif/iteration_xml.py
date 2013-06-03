#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
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
        pass


def create_new_iteration_xml_file(iteration_name, solver_name, events):
    """
    Creates a new iteration file.

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
        E.rejection_criteria(""),
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
