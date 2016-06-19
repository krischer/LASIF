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

    def __init__(self, iteration_xml_filename, stf_fct):
        """
        Init function takes a Iteration XML file and the function to
        calculate the source time function..
        """
        if not os.path.exists(iteration_xml_filename):
            msg = "File '%s' not found." % iteration_xml_filename
            raise ValueError(msg)
        self._parse_iteration_xml(iteration_xml_filename)
        self.stf_fct = stf_fct

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.__dict__ == other.__dict__:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def _parse_iteration_xml(self, iteration_xml_filename):
        """
        Parses the given iteration xml file and stores the information with the
        class instance.
        """
        root = etree.parse(iteration_xml_filename).getroot()

        # The iteration name is dependent on the filename.
        self.iteration_name = re.sub(r"\.xml$", "", re.sub(
            r"^ITERATION_", "", os.path.basename(iteration_xml_filename)))

        self.description = \
            self._get_if_available(root, "iteration_description")
        self.comments = [_i.text for _i in root.findall("comment") if _i.text]

        self.scale_data_to_synthetics = \
            self._get_if_available(root, "scale_data_to_synthetics")

        # Defaults to True.
        if self.scale_data_to_synthetics is None:
            self.scale_data_to_synthetics = True
        elif self.scale_data_to_synthetics.lower() == "true":
            self.scale_data_to_synthetics = True
        elif self.scale_data_to_synthetics.lower() == "false":
            self.scale_data_to_synthetics = False
        else:
            raise ValueError("Value '%s' invalid for "
                             "'scale_data_to_synthetics'." %
                             self.scale_data_to_synthetics)

        self.data_preprocessing = {}
        prep = root.find("data_preprocessing")
        self.data_preprocessing["highpass_period"] = \
            float(self._get(prep, "highpass_period"))
        self.data_preprocessing["lowpass_period"] = \
            float(self._get(prep, "lowpass_period"))

        self.solver_settings = \
            _recursive_dict(root.find("solver_parameters"))[1]

        self.events = OrderedDict()
        for event in root.findall("event"):
            event_name = self._get(event, "event_name")
            comments = [_i.text for _i in event.findall("comment") if _i.text]
            self.events[event_name] = {
                "event_weight": float(self._get(event, "event_weight")),
                "stations": OrderedDict(),
                "comments": comments}
            for station in event.findall("station"):
                station_id = self._get(station, "station_id")
                comments = [_i.text
                            for _i in station.findall("comment") if _i.text]
                self.events[event_name]["stations"][station_id] = {
                    "station_weight": float(self._get(station,
                                            "station_weight")),
                    "comments": comments}

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

    def _get(self, element, node_name):
        return element.find(node_name).text

    def _get_if_available(self, element, node_name):
        item = element.find(node_name)
        if item is not None:
            return item.text
        return None

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

    def write(self, filename):
        """
        Serialized the Iteration structure once again.

        :param filename: The path that will be written to.
        """
        solver_settings = _recursive_etree(
            self.solver_settings["solver_settings"])

        contents = []
        contents.append(E.iteration_name(self.iteration_name))

        if self.description:
            contents.append(E.iteration_description(self.description))

        if self.comments:
            contents.extend([E.comment(_i) for _i in self.comments])

        contents.extend([
            E.scale_data_to_synthetics(str(
                self.scale_data_to_synthetics).lower()),
            E.data_preprocessing(
                E.highpass_period(
                    str(self.data_preprocessing["highpass_period"])),
                E.lowpass_period(
                    str(self.data_preprocessing["lowpass_period"]))
            ),
            E.solver_parameters(
                E.solver(self.solver_settings["solver"]),
                E.solver_settings(*solver_settings)
            ),
        ])

        # Add all events.
        for key, value in self.events.iteritems():
            event = E.event(
                E.event_name(key),
                E.event_weight(str(value["event_weight"])),
                *[E.comment(_i) for _i in value["comments"] if _i]
            )
            for station_id, station_value in value["stations"].iteritems():
                event.append(E.station(
                    E.station_id(station_id),
                    E.station_weight(str(station_value["station_weight"])),
                    *[E.comment(_i)
                      for _i in station_value["comments"] if _i]
                ))
            contents.append(event)

        doc = E.iteration(*contents)
        doc.getroottree().write(filename, xml_declaration=True,
                                pretty_print=True, encoding="UTF-8")


def _recursive_dict(element):
    """
    Helper function to create a dictionary from an etree element.
    """
    if element.tag == "relaxation_parameter_list":
        tau = [float(_i.text) for _i in element.findall("tau")]
        w = [float(_i.text) for _i in element.findall("w")]
        return "relaxation_parameter_list", {"tau": tau, "w": w}
    text = element.text
    try:
        text = int(text)
    except:
        try:
            text = float(text)
        except:
            pass
    if isinstance(text, basestring):
        if text.lower() == "false":
            text = False
        elif text.lower() == "true":
            text = True
    return element.tag, \
        OrderedDict(map(_recursive_dict, element)) or text


def _recursive_etree(dictionary):
    """
    Helper function to create a list of etree elements from a dict.
    """
    from collections import OrderedDict
    import itertools

    contents = []
    for key, value in dictionary.iteritems():
        if key == "relaxation_parameter_list":
            # Wild iterator to arrive at the desired etree. If somebody else
            # ever reads this just look at the output and do it some other
            # way...
            contents.append(E.relaxation_parameter_list(
                *[getattr(E, i[0])(str(i[1][1]), number=str(i[1][0]))
                  for i in itertools.chain(*itertools.izip(
                      itertools.izip_longest(
                          [], enumerate(value["tau"]), fillvalue="tau"),
                      itertools.izip_longest(
                          [], enumerate(value["w"]), fillvalue="w")))]
            ))
            continue
        if isinstance(value, OrderedDict):
            contents.append(getattr(E, key)(*_recursive_etree(value)))
            continue
        if value is True:
            value = "true"
        elif value is False:
            value = "false"
        contents.append(getattr(E, key)(str(value)))
    return contents


def create_iteration_xml_string(iteration_name, solver_name, events,
                                min_period, max_period, quiet=False):
    """
    Creates a new iteration string.

    Returns a string containing the XML representation.

    :param iteration_name: The iteration name. Should start with a number.
    :param solver_name: The name of the solver to be used for this iteration.
    :param events: A dictionary. The key is the event name, and the value a
        list of station ids for this event.
    :param min_period: The minimum period for the iteration.
    :param max_period: The maximum period for the iteration.
    :param quiet: Do not print anything if set to `True`.
    """
    solver_doc = _get_default_solver_settings(solver_name, min_period,
                                              max_period, quiet=quiet)
    if solver_name.lower() == "ses3d_4_1":
        solver_name = "SES3D 4.1"
    elif solver_name.lower() == "ses3d_2_0":
        solver_name = "SES3D 2.0"
    elif solver_name.lower() == "specfem3d_cartesian":
        solver_name = "SPECFEM3D CARTESIAN"
    elif solver_name.lower() == "specfem3d_globe_cem":
        solver_name = "SPECFEM3D GLOBE CEM"
    else:
        raise NotImplementedError

    if min_period >= max_period:
        msg = "min_period needs to be smaller than max_period."
        raise ValueError(msg)

    # Loop over all events.
    events_doc = []
    # Also over all stations.
    for event_name, stations in events.iteritems():
        stations_doc = [E.station(
            E.station_id(station),
            E.station_weight("1.0")) for station in stations]
        events_doc.append(E.event(
            E.event_name(event_name),
            E.event_weight("1.0"),
            *stations_doc))

    doc = E.iteration(
        E.iteration_name(iteration_name),
        E.iteration_description(""),
        E.comment(""),
        E.scale_data_to_synthetics("true"),
        E.data_preprocessing(
            E.highpass_period(str(max_period)),
            E.lowpass_period(str(min_period))),
        E.solver_parameters(
            E.solver(solver_name),
            solver_doc),
        *events_doc)

    string_doc = etree.tostring(doc, pretty_print=True,
                                xml_declaration=True, encoding="UTF-8")
    return string_doc


def _get_default_solver_settings(solver, min_period, max_period, quiet=False):
    """
    Helper function returning etree representation of a solver's default
    settings.

    :param quiet: Do not print anything if set to `True`.
    """
    known_solvers = ["ses3d_4_1", "ses3d_2_0", "specfem3d_cartesian",
                     "specfem3d_globe_cem"]
    if solver.lower() == "ses3d_4_1":
        from lasif.tools import Q_discrete
        from lasif.utils import generate_ses3d_4_1_template

        # Generate the relaxation weights for SES3D.
        w_p, tau_p = Q_discrete.calculate_Q_model(
            N=3,
            # These are suitable for the default frequency range.
            f_min=1.0 / max_period,
            f_max=1.0 / min_period,
            iterations=10000,
            initial_temperature=0.1,
            cooling_factor=0.9998, quiet=quiet)

        return generate_ses3d_4_1_template(w_p, tau_p)
    elif solver.lower() == "ses3d_2_0":
        from lasif.utils import generate_ses3d_2_0_template
        return generate_ses3d_2_0_template()
    elif solver.lower() == "specfem3d_cartesian":
        from lasif.utils import generate_specfem3d_cartesian_template
        return generate_specfem3d_cartesian_template()
    elif solver.lower() == "specfem3d_globe_cem":
        from lasif.utils import generate_specfem3d_globe_cem_template
        return generate_specfem3d_globe_cem_template()
    else:
        msg = "Solver '%s' not known. Known solvers: %s" % (
            solver, ",".join(known_solvers))
        raise LASIFError(msg)
