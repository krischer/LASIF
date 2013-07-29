#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some utility functionality.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from fnmatch import fnmatch
from lxml.builder import E


def channel_in_parser(parser_object, channel_id, starttime, endtime):
    """
    Simply function testing if a given channel is part of a Parser object.

    Returns True or False.

    :type parser_object: :class:`obspy.xseed.Parser`
    :param parser_object: The parser object.
    """
    channels = parser_object.getInventory()["channels"]
    for chan in channels:
        if not fnmatch(chan["channel_id"], channel_id):
            continue
        if starttime < chan["start_date"]:
            continue
        if chan["end_date"] and \
                (endtime > chan["end_date"]):
            continue
        return True
    return False


def table_printer(header, data):
    """
    Pretty table printer.

    :type header: A list of strings
    :param data: A list of lists containing data items.
    """
    row_format = "{:>15}" * (len(header))
    print row_format.format(*(["=" * 15] * len(header)))
    print row_format.format(*header)
    print row_format.format(*(["=" * 15] * len(header)))
    for row in data:
        print row_format.format(*row)


def recursive_dict(element):
    """
    Maps an XML tree into a dict of dict.

    From the lxml documentation.
    """
    return element.tag, \
        dict(map(recursive_dict, element)) or element.text


def generate_ses3d_4_0_template():
    """
    Generates a template for SES3D input files.

    Returns the etree representation.
    """
    doc = E.solver_settings(
        E.simulation_parameters(
            E.number_of_time_steps("500"),
            E.time_increment("0.75"),
            E.is_dissipative("false")),
        E.output_directory("../OUTPUT/CHANGE_ME/{{EVENT_NAME}}"),
        E.adjoint_output_parameters(
            E.sampling_rate_of_forward_field("10"),
            E.forward_field_output_directory(
                "../OUTPUT/CHANGE_ME/ADJOINT/{{EVENT_NAME}}")),
        E.computational_setup(
            E.nx_global("15"),
            E.ny_global("15"),
            E.nz_global("10"),
            E.lagrange_polynomial_degree("4"),
            E.px_processors_in_theta_direction("1"),
            E.py_processors_in_phi_direction("1"),
            E.pz_processors_in_r_direction("1")),
        E.relaxation_parameter_list(
            E.tau("1.7360", number="0"),
            E.w("2.5133", number="0"),
            E.tau("14.6211", number="1"),
            E.w("2.4089", number="1"),
            E.tau("13.7054", number="2"),
            E.w("0.1005", number="2")))

    return doc
