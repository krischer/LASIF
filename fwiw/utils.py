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
from lxml import etree
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
        if chan["channel_id"] != channel_id:
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


def generate_ses3d_template(filename):
    """
    Generates a template for SES3D input files.

    :param filename: Where to store it.
    """
    doc = E.ses3d_4_0_input_file_template(
        E.simulation_parameters(
            E.number_of_timesteps("4000"),
            E.time_increment("0.13")),
        E.output_directory("../DATA/OUTPUT/CHANGE_ME/"),
        E.adjoint_output_parameters(
            E.sampling_rate_of_forward_field("15"),
            E.forward_field_output_directory("../DATA/OUTPUT/CHANGE_ME/")),
        E.computational_setup(
            E.nx_global("66"),
            E.ny_global("108"),
            E.nz_global("28"),
            E.lagrange_polynomial_degree("4"),
            E.px_processors_in_theta_direction("3"),
            E.py_processors_in_theta_direction("4"),
            E.pz_processors_in_theta_direction("4")))
    string_doc = etree.tostring(doc, pretty_print=True,
        xml_declaration=True, encoding="UTF-8")

    with open(filename, "wb") as open_file:
        open_file.write(string_doc)
