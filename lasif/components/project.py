#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project components class.

It is important to not import necessary things at the method level to make
importing this file as fast as possible. Otherwise using the command line
interface feels sluggish and slow. Import things only the functions they are
needed.

:copyright: Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013

:license: GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import glob
import imp
import inspect
import os
import warnings

from lasif import LASIFError, LASIFNotFoundError, LASIFWarning
import lasif.domain

from .actions import ActionsComponent
from .adjoint_sources import AdjointSourcesComponent
from .communicator import Communicator
from .component import Component
from .downloads import DownloadsComponent
from .events import EventsComponent
from .iterations import IterationsComponent
from .kernels import KernelsComponent
from .models import ModelsComponent
from .query import QueryComponent
from .stations import StationsComponent
from .validator import ValidatorComponent
from .visualizations import VisualizationsComponent
from .waveforms import WaveformsComponent
from .windows import WindowsAndAdjointSourcesComponent


class Project(Component):
    """
    A class managing LASIF projects.

    It represents the heart of LASIF.
    """
    def __init__(self, project_root_path, init_project=False):
        """
        Upon intialization, set the paths and read the config file.

        :type project_root_path: str
        :param project_root_path: The root path of the project.
        :type init_project: str
        :param init_project: Determines whether or not to initialize a new
            project, e.g. create the necessary folder structure. If a string is
            passed, the project will be given this name. Otherwise a default
            name will be chosen. Defaults to False.
        """
        # Setup the paths.
        self.__setup_paths(project_root_path)
        # Manually create the CACHE folder - it is needed for initializing
        # some components and thus cannot use the folder creation logic for
        # the rest of the project structure.
        if not os.path.exists(self.paths["cache"]):
            os.makedirs(self.paths["cache"])

        if init_project:
            if not os.path.exists(project_root_path):
                os.makedirs(project_root_path)
            self.__init_new_project(init_project)
            return

        if not os.path.exists(self.paths["config_file"]):
            msg = ("Could not find the project's config file. Wrong project "
                   "path or uninitialized project?")
            raise LASIFError(msg)

        self.__project_function_cache = {}

        # Setup the communicator and register this component.
        self.__comm = Communicator()
        super(Project, self).__init__(self.__comm, "project")

        self.__setup_components()

        # Finally update the folder structure.
        self.__update_folder_structure()

        self._read_config_file()

        self.__copy_fct_templates(init_project=init_project)

    def __str__(self):
        """
        Pretty string representation.
        """
        # Count all files and sizes.
        ret_str = "LASIF project \"%s\"\n" % self.config["name"]
        ret_str += "\tDescription: %s\n" % self.config["description"]
        ret_str += "\tProject root: %s\n" % self.paths["root"]
        ret_str += "\tContent:\n"
        ret_str += "\t\t%i events\n" % self.comm.events.count()

        d = str(self.domain)
        ret_str += "\n".join(["\t" + i for i in d.splitlines()])

        return ret_str

    def __copy_fct_templates(self, init_project):
        """
        Copies the function templates to the project folder if they do not
        yet exist.

        :param init_project: Flag if this is called during the project
            initialization or not. If not called during project initialization
            this function will raise a warning to make users aware of the
            changes in LASIF.
        """
        directory = os.path.abspath(os.path.join(
            os.path.dirname(inspect.getfile(
                inspect.currentframe())),
            os.path.pardir,
            "function_templates"))
        for template_filename in glob.glob(os.path.join(directory, "*.py")):
            filename = os.path.basename(template_filename)
            new_filename = os.path.join(self.paths["functions"], filename)
            if not os.path.exists(new_filename):
                if not init_project:
                    warnings.warn(
                        "Function template '%s' did not exist. It does now. "
                        "Did you update a later LASIF version? Please make "
                        "sure you are aware of the changes." % filename,
                        LASIFWarning)
                import shutil
                shutil.copy(src=template_filename, dst=new_filename)

    def _read_config_file(self):
        """
        Parse the config file.
        """

    def get_communicator(self):
        return self.__comm

    def __setup_components(self):
        """
        Setup the different components of the project. The goal is to
        decouple them as much as possible to keep the structure sane and
        maintainable.

        Communication will happen through the communicator which will also
        keep the references to the single components.
        """
        # Basic components.
        EventsComponent(folder=self.paths["events"], communicator=self.comm,
                        component_name="events")
        WaveformsComponent(data_folder=self.paths["data"],
                           synthetics_folder=self.paths["synthetics"],
                           communicator=self.comm, component_name="waveforms")
        ModelsComponent(models_folder=self.paths["models"],
                        communicator=self.comm,
                        component_name="models")
        KernelsComponent(kernels_folder=self.paths["kernels"],
                         communicator=self.comm,
                         component_name="kernels")
        IterationsComponent(iterations_folder=self.paths["iterations"],
                            communicator=self.comm,
                            component_name="iterations")

        # Action and query components.
        QueryComponent(communicator=self.comm, component_name="query")
        VisualizationsComponent(communicator=self.comm,
                                component_name="visualizations")
        ActionsComponent(communicator=self.comm,
                         component_name="actions")
        ValidatorComponent(communicator=self.comm,
                           component_name="validator")

        WindowsAndAdjointSourcesComponent(
            folder=self.paths["windows_and_adjoint_sources"],
            communicator=self.comm,
            component_name="wins_and_adj_sources")
        # # Window and adjoint source components.
        # WindowsAndAdjointSourcesComponent(
        #     folder=self.paths["windows_and_adjoint_sources"],
        #     communicator=self.comm,
        #     component_name="win_adjoint")
        # AdjointSourcesComponent(ad_src_folder=self.paths["adjoint_sources"],
        #                         communicator=self.comm,
        #                         component_name="adjoint_sources")

        # Data downloading component.
        DownloadsComponent(communicator=self.comm,
                           component_name="downloads")

    def __setup_paths(self, root_path):
        """
        Central place to define all paths.
        """
        # Every key containing the string "file" denotes a file, all others
        # should denote directories.
        self.paths = {}
        self.paths["root"] = root_path

        self.paths["events"] = os.path.join(root_path, "EVENTS")
        self.paths["data"] = os.path.join(root_path, "DATA")
        self.paths["cache"] = os.path.join(root_path, "CACHE")
        self.paths["logs"] = os.path.join(root_path, "LOGS")
        self.paths["models"] = os.path.join(root_path, "MODELS")
        self.paths["wavefields"] = os.path.join(root_path, "WAVEFIELDS")
        self.paths["iterations"] = os.path.join(root_path, "ITERATIONS")
        self.paths["synthetics"] = os.path.join(root_path, "SYNTHETICS")
        self.paths["kernels"] = os.path.join(root_path, "KERNELS")
        self.paths["output"] = os.path.join(root_path, "OUTPUT")
        # Path for the custom functions.
        self.paths["functions"] = os.path.join(root_path, "FUNCTIONS")

        # Paths for various files.
        self.paths["config_file"] = os.path.join(root_path,
                                                 "lasif_config.toml")

        self.paths["windows_and_adjoint_sources"] = os.path.join(
            root_path, "ADJOINT_SOURCES_AND_WINDOWS")

    def __update_folder_structure(self):
        """
        Updates the folder structure of the project.
        """
        for name, path in self.paths.items():
            if "file" in name or os.path.exists(path):
                continue
            os.makedirs(path)
        events = self.comm.events.list()
        for event in events:
            event_folder = os.path.join(self.paths["data"], event)
            if os.path.exists(event_folder):
                continue
            os.makedirs(event_folder)

    def __init_new_project(self, project_name):
        """
        Initializes a new project. This currently just means that it creates a
        default config file. The folder structure is checked and rebuilt every
        time the project is initialized anyways.
        """
        import toml

        if not project_name:
            project_name = "LASIFProject"

        config = {"lasif_project": {
            "project_name": project_name,
            "description": "",
            "mesh_file": "",
            "download_settings": {
                "seconds_before_event": 300.0,
                "seconds_after_event": 3600.0,
                "interstation_distance_in_meters": 1000.0,
                "channel_priorities": ["BH[Z,N,E]", "LH[Z,N,E]", "HH[Z,N,E]",
                                       "EH[Z,N,E]", "MH[Z,N,E]"],
                "location_priorities": ("", "00", "10", "20", "01", "02")
        }}}

        with open(self.paths["config_file"], "w") as fh:
            toml.dump(config, fh)

    def get_project_function(self, fct_type):
        """
        Helper importing the project specific function.

        :param fct_type: The desired function.
        """
        # Cache to avoid repeated imports.
        if fct_type in self.__project_function_cache:
            return self.__project_function_cache[fct_type]

        # type / filename map
        fct_type_map = {
            "window_picking_function": "window_picking_function.py",
            "preprocessing_function": "preprocessing_function.py",
            "process_synthetics": "process_synthetics.py",
            "source_time_function": "source_time_function.py"
        }

        if fct_type not in fct_type:
            msg = "Function '%s' not found. Available types: %s" % (
                fct_type, str(list(fct_type_map.keys())))
            raise LASIFNotFoundError(msg)

        filename = os.path.join(self.paths["functions"],
                                fct_type_map[fct_type])
        if not os.path.exists(filename):
            msg = "No file '%s' in existence." % filename
            raise LASIFNotFoundError(msg)

        fct_template = imp.load_source("_lasif_fct_template", filename)
        try:
            fct = getattr(fct_template, fct_type)
        except AttributeError:
            raise LASIFNotFoundError("Could not find function %s in file '%s'"
                                     % (fct_type, filename))

        if not callable(fct):
            raise LASIFError("Attribute %s in file '%s' is not a function."
                             % (fct_type, filename))

        # Add to cache.
        self.__project_function_cache[fct_type] = fct
        return fct

    def get_output_folder(self, type, tag):
        """
        Generates a output folder in a unified way.

        :param type: The type of data. Will be a subfolder.
        :param tag: The tag of the folder. Will be postfix of the final folder.
        """
        from obspy import UTCDateTime
        d = str(UTCDateTime()).replace(":", "-").split(".")[0]

        output_dir = os.path.join(self.paths["output"], type.lower(),
                                  "%s__%s" % (d, tag))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def get_log_file(self, log_type, description):
        """
        Returns the name of a log file. It will create all necessary
        directories along the way but not the log file itsself.

        :param log_type: The type of logging. Will result in a subfolder.
            Examples for this are ``"PROCESSING"``, ``"DOWNLOADS"``, ...
        :param description: Short description of what is being downloaded.
            Will be used to derive the name of the logfile.
        """
        from obspy import UTCDateTime
        log_dir = os.path.join(self.paths["logs"], log_type)
        filename = ("%s___%s" % (str(UTCDateTime()), description))
        filename += os.path.extsep + "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return os.path.join(log_dir, filename)
