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
from collections import namedtuple
from geographiclib import geodesic
from fnmatch import fnmatch
from lxml.builder import E

from lasif import LASIFNotFoundError


def is_mpi_env():
    """
    Returns True if currently in an MPI environment.
    """
    from mpi4py import MPI
    if MPI.COMM_WORLD.size == 1 and MPI.COMM_WORLD.rank == 0:
        return False
    return True


def channel_in_parser(parser_object, channel_id, starttime, endtime):
    """
    Simply function testing if a given channel is part of a Parser object.

    Returns True or False.

    :type parser_object: :class:`obspy.io.xseed.Parser`
    :param parser_object: The parser object.
    """
    channels = parser_object.get_inventory()["channels"]
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


def generate_ses3d_4_1_template(w_p, tau_p):
    """
    Generates a template for SES3D 4.1 input files.

    Returns the etree representation.
    """
    if len(tau_p) != 3 or len(w_p) != 3:
        msg = "SES3D currently only supports three superimposed linear solids."
        raise ValueError(msg)
    doc = E.solver_settings(
        E.simulation_parameters(
            E.number_of_time_steps("500"),
            E.time_increment("0.75"),
            E.is_dissipative("true")),
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
            E.tau(str(tau_p[0]), number="0"),
            E.w(str(w_p[0]), number="0"),
            E.tau(str(tau_p[1]), number="1"),
            E.w(str(w_p[1]), number="1"),
            E.tau(str(tau_p[2]), number="2"),
            E.w(str(w_p[2]), number="2")))

    return doc


def generate_ses3d_2_0_template():
    """
    Generates a template for SES3D 2.0 input files.

    Returns the etree representation.
    """
    doc = E.solver_settings(
        E.simulation_parameters(
            E.number_of_time_steps("500"),
            E.time_increment("0.75"),
            E.is_dissipative("true")),
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
            E.pz_processors_in_r_direction("1")))

    return doc


def generate_specfem3d_cartesian_template():
    """
    Generates a template for SPECFEM3D CARTESIAN input files.

    Returns the etree representation.
    """
    doc = E.solver_settings(
        E.simulation_parameters(
            E.number_of_time_steps("500"),
            E.time_increment("0.75")),
        E.output_directory("../OUTPUT/CHANGE_ME/{{EVENT_NAME}}"),
        E.computational_setup(
            E.number_of_processors("128")))

    return doc


def generate_specfem3d_globe_cem_template():
    """
    Generates a template for SPECFEM3D GLOBE CEM input files.

    Returns the etree representation.
    """
    doc = E.solver_settings(
        E.simulation_parameters(
            E.number_of_time_steps("2000"),
            E.time_increment("0.1")),
        E.local_path("../OUTPUT/CHANGE_ME/{{EVENT_NAME}}"),
        E.local_temp_path("../OUTPUT/CHANGE_ME/{{EVENT_NAME}}"),
        E.adjoint_source_time_shift("-10"),
        E.computational_setup(
            E.number_of_processors_xi("5"),
            E.number_of_processors_eta("5"),
            E.number_of_chunks("1"),
            E.elements_per_chunk_xi("240"),
            E.elements_per_chunk_eta("240"),
            E.model("1D_transversely_isotropic_prem"),
            E.simulate_oceans("true"),
            E.simulate_ellipticity("true"),
            E.simulate_topography("true"),
            E.simulate_gravity("true"),
            E.simulate_rotation("true"),
            E.simulate_attenuation("true"),
            E.fast_undo_attenuation("false"),
            E.use_gpu("false")))

    return doc


def point_in_domain(latitude, longitude, domain,
                    rotation_axis=[0.0, 0.0, 1.0],
                    rotation_angle_in_degree=0.0):
    """
    Simple function checking if a geographic point is placed inside a
    rotated spherical section. It simple rotates the point and checks if it
    is inside the non-rotated domain.

    Domain is a dictionary containing at least the following keys:
        * "minimum_latitude"
        * "maximum_latitude"
        * "minimum_longitude"
        * "maximum_longitude"
        * "boundary_width_in_degree"

    Returns True or False.
    """
    from lasif import rotations
    min_latitude = domain["minimum_latitude"] + \
        domain["boundary_width_in_degree"]
    max_latitude = domain["maximum_latitude"] - \
        domain["boundary_width_in_degree"]
    min_longitude = domain["minimum_longitude"] + \
        domain["boundary_width_in_degree"]
    max_longitude = domain["maximum_longitude"] - \
        domain["boundary_width_in_degree"]

    # Rotate the station and check if it is still in bounds.
    r_lat, r_lng = rotations.rotate_lat_lon(
        latitude, longitude, rotation_axis, -1.0 * rotation_angle_in_degree)
    # Check if in bounds. If not continue.
    if not (min_latitude <= r_lat <= max_latitude) or \
            not (min_longitude <= r_lng <= max_longitude):
        return False
    return True


def sizeof_fmt(num):
    """
    Handy formatting for human readable filesize.

    From http://stackoverflow.com/a/1094933/1657047
    """
    for x in ["bytes", "KB", "MB", "GB"]:
        if num < 1024.0 and num > -1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, "TB")


Point = namedtuple("Point", ["lat", "lng"])


def greatcircle_points(point_1, point_2, max_extension=None,
                       max_npts=3000):
    """
    Generator yielding a number points along a greatcircle from point_1 to
    point_2. Max extension is the normalization factor. If the distance between
    point_1 and point_2 is exactly max_extension, then 3000 points will be
    returned, otherwise a fraction will be returned.

    If max_extension is not given, the generator will yield exactly max_npts
    points.
    """
    point = geodesic.Geodesic.WGS84.Inverse(
        lat1=point_1.lat, lon1=point_1.lng, lat2=point_2.lat,
        lon2=point_2.lng)
    line = geodesic.Geodesic.WGS84.Line(
        point_1.lat, point_1.lng, point["azi1"])

    if max_extension:
        npts = int((point["a12"] / float(max_extension)) * max_npts)
    else:
        npts = max_npts - 1
    if npts == 0:
        npts = 1
    for i in xrange(npts + 1):
        line_point = line.Position(i * point["s12"] / float(npts))
        yield Point(line_point["lat2"], line_point["lon2"])


def channel2station(value):
    """
    Helper function converting a channel id to a station id. Will not change
    a passed station id.

    :param value: The channel id as a string.

    >>> channel2station("BW.FURT.00.BHZ")
    'BW.FURT'
    >>> channel2station("BW.FURT")
    'BW.FURT'
    """
    return ".".join(value.split(".")[:2])


def select_component_from_stream(st, component):
    """
    Helper function selecting a component from a Stream an raising the proper
    error if not found.

    This is a bit more flexible then stream.select() as it works with single
    letter channels and lowercase channels.
    """
    component = component.upper()
    component = [tr for tr in st if tr.stats.channel[-1].upper() == component]
    if not component:
        raise LASIFNotFoundError("Component %s not found in Stream." %
                                 component)
    elif len(component) > 1:
        raise LASIFNotFoundError("More than 1 Trace with component %s found "
                                 "in Stream." % component)
    return component[0]


def get_event_filename(event, prefix):
    """
    Helper function generating a descriptive event filename.

    :param event: The event object.
    :param prefix: A prefix for the file, denoting e.g. the event catalog.

    >>> from obspy import read_events
    >>> event = read_events()[0]
    >>> print get_event_filename(event, "GCMT")
    GCMT_event_KYRGYZSTAN-XINJIANG_BORDER_REG._Mag_4.4_2012-4-4-14.xml
    """
    from obspy.geodetics import FlinnEngdahl

    mag = event.preferred_magnitude() or event.magnitudes[0]
    org = event.preferred_origin() or event.origins[0]

    # Get the flinn_engdahl region for a nice name.
    fe = FlinnEngdahl()
    region_name = fe.get_region(org.longitude, org.latitude)
    region_name = region_name.replace(" ", "_")
    # Replace commas, as some file systems cannot deal with them.
    region_name = region_name.replace(",", "")

    return "%s_event_%s_Mag_%.1f_%s-%s-%s-%s.xml" % \
        (prefix, region_name, mag.mag, org.time.year, org.time.month,
         org.time.day, org.time.hour)
