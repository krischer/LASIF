#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import glob
import os

from .component import Component
import shutil


class IterationsComponent(Component):
    """
    Component dealing with the iteration xml files.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, communicator, component_name):
        self.__cached_iterations = {}
        super(IterationsComponent, self).__init__(communicator,
                                                  component_name)

    def get_long_iteration_name(self, iteration_name):
        """
        Returns the long form of an iteration from its short or long name.

        >>> comm = getfixture('iterations_comm')
        >>> comm.iterations.get_long_iteration_name("1")
        'ITERATION_1'
        """
        if iteration_name[:10] == "ITERATION_":
            iteration_name = iteration_name[10:]
        return "ITERATION_%s" % iteration_name

    def write_info_toml(self, iteration_name, simulation_type):
        """
        Write a toml file to store information related to how the important
        config settings were when input files were generated. This will
        create a new file when forward input files are generated.
        :param iteration_name: The iteration for which to write into toml
        :param simulation_type: The type of simulation.
        """
        settings = self.comm.project.solver_settings
        project_info = self.comm.project.config
        data_proc = self.comm.project.processing_params
        info_path = os.path.join(self.comm.project.paths["synthetics"],
                                 "INFORMATION",
                                 self.get_long_iteration_name(iteration_name))

        toml_string = f"# Information to store how things were in " \
                      f"this iteration.\n \n" \
                      f"[parameters]\n" \
                      f"    number_of_absorbing_layers = " \
                      f"{settings['number_of_absorbing_layers']} \n" \
                      f"    salvus_start_time = " \
                      f"{- settings['time_increment']} \n" \
                      f"    end_time = {settings['end_time']} \n" \
                      f"    time_increment = " \
                      f"{settings['time_increment']}\n" \
                      f"    polynomial_order = " \
                      f"{settings['polynomial_order']}\n" \
                      f"    with_anisotropy = " \
                      f"\"{settings['with_anisotropy']}\"\n" \
                      f"    with_attenuation = " \
                      f"\"{settings['with_attenuation']}\"\n" \
                      f"    stf = " \
                      f"\"{settings['source_time_function_type']}\" " \
                      f"# source time function type\n" \
                      f"    mesh_file = " \
                      f"\"{project_info['mesh_file']}\"\n" \
                      f"    highpass_period = " \
                      f"{data_proc['highpass_period']}\n" \
                      f"    lowpass_period = " \
                      f"{data_proc['lowpass_period']}\n"

        if simulation_type == "adjoint":
            toml_string += f"    misfit_type = " \
                           f"\"{project_info['misfit_type']}\"\n"

        info_file = os.path.join(info_path, f"{simulation_type}.toml")

        with open(info_file, "w") as fh:
            fh.write(toml_string)
        print(f"Information about input files stored in {info_file}")

    def setup_directories_for_iteration(self, iteration_name,
                                        remove_dirs=False):
        """
        Sets up the directory structure required for the iteration
        :param iteration_name: The iteration for which to create the folders.
        :param remove_dirs: Boolean if set to True the iteration is removed
        """
        long_iter_name = self.get_long_iteration_name(iteration_name)
        self._create_synthetics_folder_for_iteration(long_iter_name,
                                                     remove_dirs)
        self._create_input_files_folder_for_iteration(long_iter_name,
                                                      remove_dirs)
        self._create_adjoint_sources_and_windows_folder_for_iteration(
            long_iter_name, remove_dirs)
        self._create_model_folder_for_iteration(long_iter_name, remove_dirs)
        self._create_iteration_folder_for_iteration(long_iter_name,
                                                    remove_dirs)

    def setup_iteration_toml(self, iteration_name, remove_dirs=False):
        """
        Sets up a toml file which can be used to keep track of needed
        information related to the iteration. It can be used to specify which
        events to use and it can remember which input parameters were used.
        :param iteration_name: The iteration for which to create the folders.
        :param remove_dirs: Boolean if set to True the iteration is removed
        """

        long_iter_name = self.get_long_iteration_name(iteration_name)

        path = self.comm.project.paths["iterations"]
        syn_folder = self.comm.project.paths["synthetics"]
        sim_folder = os.path.join(syn_folder, "INFORMATION", long_iter_name)
        file = os.path.join(path, long_iter_name, "central_info.toml")
        event_file = os.path.join(path, long_iter_name, "events_used.toml")
        forward_file = os.path.join(sim_folder, "forward.toml")
        adjoint_file = os.path.join(sim_folder, "adjoint.toml")
        step_file = os.path.join(sim_folder, "step_length.toml")

        toml_string = f"# This toml file includes information relative to " \
                      f"this iteration: {iteration_name}. \n" \
                      f"# It contains direct information as well as paths " \
                      f"to other toml files with other information.\n \n" \
                      f"[events]\n" \
                      f"    # In this file you can modify the used events " \
                      f"in the iteration. \n    # This is what your " \
                      f"commands will read when you don't specify events.\n" \
                      f"    events_used = \"{event_file}\"\n\n" \
                      f"[simulations]\n" \
                      f"    # These files will be created or updated every " \
                      f"time" \
                      f" you generate input files for the respective " \
                      f"simulations.\n" \
                      f"    forward = \"{forward_file}\"\n" \
                      f"    adjoint = \"{adjoint_file}\"\n" \
                      f"    step_length = \"{step_file}\"\n\n" \
                      f"    # That's it, if you need more, contact " \
                      f"developers.\n" \

        with open(file, "w") as fh:
            fh.write(toml_string)
        print(f"Information about iteration stored in {file}")

    def setup_events_toml(self, iteration_name, remove_dirs=False):
        """
        Writes all events into a toml file. User can modify this if he wishes
        to use less events for this specific iteration. Lasif should be smart
        enough to know which events were used in which iteration.
        """
        long_iter_name = self.get_long_iteration_name(iteration_name)
        path = self.comm.project.paths["iterations"]

        event_file = os.path.join(path, long_iter_name, "events_used.toml")

        toml_string = "# Here we store information regarding which events " \
                      "are " \
                      "used \n# User can remove events at will and Lasif " \
                      "should recognise it when input files are generated.\n" \
                      "# Everything related to using all events, should " \
                      "read this file and classify that as all events for " \
                      "iteration.\n\n" \
                      "[events]\n" \
                      "    events_used = ["
        s = 0
        for event in self.comm.events.list():
            if s == len(self.comm.events.list()) - 1:
                toml_string += "'" + event + "']"
            else:
                toml_string += "'" + event + "',\n"
            s += 1

        with open(event_file, "w") as fh:
            fh.write(toml_string)

    def _create_iteration_folder_for_iteration(self, long_iteration_name,
                                               remove_dirs=False):
        """
        Create folder for this iteration in the iteration information directory
        """

        path = self.comm.project.paths["iterations"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if remove_dirs:
            shutil.rmtree(folder)

    def _create_synthetics_folder_for_iteration(self, long_iteration_name,
                                                remove_dirs=False):
        """
        Create the synthetics folder if it does not yet exist.
        :param iteration_name: The iteration for which to create the folders.
        """

        path = self.comm.project.paths["synthetics"]

        folder_eq = os.path.join(path, "EARTHQUAKES", long_iteration_name)
        folder_info = os.path.join(path, "INFORMATION", long_iteration_name)
        if not os.path.exists(folder_eq):
            os.makedirs(folder_eq)
        if not os.path.exists(folder_info):
            os.makedirs(folder_info)
        if remove_dirs:
            shutil.rmtree(folder_eq)
            shutil.rmtree(folder_info)

    def _create_input_files_folder_for_iteration(self, long_iteration_name,
                                                 remove_dirs=False):
        """
        Create the synthetics folder if it does not yet exist.
        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["salvus_input"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if remove_dirs:
            shutil.rmtree(folder)

    def _create_adjoint_sources_and_windows_folder_for_iteration(
            self, long_iteration_name, remove_dirs=False):
        """
        Create the adjoint_sources_and_windows folder if it does not yet exist.
        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["adjoint_sources"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if remove_dirs:
            shutil.rmtree(folder)

    def _create_model_folder_for_iteration(
            self, long_iteration_name, remove_dirs=False):
        """
        Create the model folder if it does not yet exist.
        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["models"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if remove_dirs:
            shutil.rmtree(folder)

    def _create_gradients_folder_for_iteration(
            self, long_iteration_name, remove_dirs=False):
        """
        Create the kernel folder if it does not yet exist.
        :param iteration_name: The iteration for which to create the folders.
        """
        path = self.comm.project.paths["gradients"]

        folder = os.path.join(path, long_iteration_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if remove_dirs:
            shutil.rmtree(folder)

    def list(self):
        """
        Returns a list of all the iterations known to LASIF.
        """
        files = [os.path.abspath(_i) for _i in glob.iglob(os.path.join(
            self.comm.project.paths["eq_synthetics"], "ITERATION_*"))]
        iterations = [os.path.basename(_i)[10:]
                      for _i in files]
        return sorted(iterations)

    def has_iteration(self, iteration_name):
        """
        Checks for existance of an iteration
        """
        iteration_name = iteration_name.lstrip("ITERATION_")
        if iteration_name in self.list():
            return True
        return False
