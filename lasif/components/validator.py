#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import colorama
import toml
import pyasdf
import os
import sys

from .component import Component


class ValidatorComponent(Component):
    """
    Component responsible for validating data inside a project. Needs access
    to a lot of functionality and should therefore be initialized fairly late.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, *args, **kwargs):
        super(ValidatorComponent, self).__init__(*args, **kwargs)
        self._reports = []
        self._total_error_count = 0
        self.files_to_be_deleted = {}

    def _print_ok_message(self):
        """
        Prints a colored OK message when a certain test has been passed.
        """
        ok_string = " %s[%sOK%s]%s" % (
            colorama.Style.BRIGHT, colorama.Style.NORMAL + colorama.Fore.GREEN,
            colorama.Fore.RESET + colorama.Style.BRIGHT,
            colorama.Style.RESET_ALL)
        print(ok_string)

    def _print_fail_message(self):
        """
        Prints a colored fail message when a certain test has been passed.
        """
        fail_string = " %s[%sFAIL%s]%s" % (
            colorama.Style.BRIGHT, colorama.Style.NORMAL + colorama.Fore.RED,
            colorama.Fore.RESET + colorama.Style.BRIGHT,
            colorama.Style.RESET_ALL)
        print(fail_string)

    def _flush_point(self):
        """
        Helper function just flushing a point to stdout to indicate progress.
        """
        sys.stdout.write(".")
        sys.stdout.flush()

    def _add_report(self, message, error_count=1):
        """
        Helper method adding a new error message.
        """
        self._reports.append(message)
        self._total_error_count += error_count

    def validate_data(self, data_and_station_file_availability=False,
                      raypaths=False):
        """
        Validates all data of the current project.

        This commands walks through all available data and checks it for
        validity.  It furthermore does some sanity checks to detect common
        problems. These should be fixed.

        Event files:
            * Validate against QuakeML 1.2 scheme.
            * Make sure they contain at least one origin, magnitude and focal
              mechanism object.
            * Check for duplicate ids amongst all QuakeML files.
            * Some simply sanity checks so that the event depth is reasonable
              and the moment tensor values as well. This is rather fragile and
              mainly intended to detect values specified in wrong units.
        """
        # Reset error and report counts.
        self._reports = []
        self._total_error_count = 0

        self._validate_event_files()

        # Assert that all waveform files have a corresponding station file.
        if data_and_station_file_availability:
            self._validate_station_and_waveform_availability()
        else:
            print("%sSkipping data and station file availability check.%s" % (
                colorama.Fore.YELLOW, colorama.Fore.RESET))

        if raypaths:
            if self.comm.project.domain.is_global_domain():
                print("%sSkipping raypath checks for global domain...%s" % (
                    colorama.Fore.YELLOW, colorama.Fore.RESET))
            else:
                self.validate_raypaths_in_domain()
        else:
            print("%sSkipping raypath checks.%s" % (
                colorama.Fore.YELLOW, colorama.Fore.RESET))

        # Depending on whether or not the tests passed, report it accordingly.
        if not self._reports:
            print("\n%sALL CHECKS PASSED%s\n"
                  "The data seems to be valid. If we missed something please "
                  "contact the developers." % (colorama.Fore.GREEN,
                                               colorama.Fore.RESET))
        else:
            folder = \
                self.comm.project.get_output_folder(
                    type="validation",
                    tag="data_integrity_report")
            filename = os.path.join(folder, "report.txt")
            seperator_string = "\n" + 80 * "=" + "\n" + 80 * "=" + "\n"
            with open(filename, "wt") as fh:
                for report in self._reports:
                    fh.write(report.strip())
                    fh.write(seperator_string)
            files_to_be_deleted_filename = \
                os.path.join(folder, "files_to_be_deleted.toml")
            with open(files_to_be_deleted_filename, "w") as fh:
                toml.dump(self.files_to_be_deleted, fh)
            print("\n%sFAILED%s\nEncountered %i errors!\n"
                  "A report has been created at '%s'.\n" %
                  (colorama.Fore.RED, colorama.Fore.RESET,
                   self._total_error_count, os.path.relpath(filename)))
            print(f"A file that can be used to clean up the project has been"
                  f"created at "
                  f"{os.path.relpath(files_to_be_deleted_filename)}\n"
                  f"It is advised to inspect the file before use.")

    def validate_raypaths_in_domain(self):
        """
        Checks that all raypaths are within the specified domain boundaries.

        Returns a list of waveform files violating that assumtion.
        """
        print("Making sure raypaths are within boundaries ", end="")

        all_good = True

        for event_name, event in self.comm.events.get_all_events().items():
            self._flush_point()
            for station_id, value in \
                    self.comm.query.get_all_stations_for_event(
                        event_name).items():

                # Check if the whole path of the event-station pair is within
                # the domain boundaries.
                if self.is_event_station_raypath_within_boundaries(
                        event_name, value["latitude"], value["longitude"],
                        raypath_steps=3):
                    continue
                all_good = False
                self.files_to_be_deleted[event_name].append(station_id)
                self._add_report(
                    f"WARNING: "
                    f"The event-station raypath for the "
                    f"station\n\t'{station_id}'\n "
                    f"does not fully lay within the domain. You might want"
                    f" to remove the file or change the domain "
                    f"specifications.")
        if all_good:
            self._print_ok_message()
        else:
            self._print_fail_message()

    def clean_up_project(self, clean_up_file):
        """

        :param clean_up_file: A toml describing the events that can be
        deleted.
        """

        clean_up_dict = toml.load(clean_up_file)
        num_of_deleted_files = 0
        for event_name, stations in clean_up_dict.items():
            filename = self.comm.waveforms.get_asdf_filename(
                event_name, data_type="raw")
            with pyasdf.ASDFDataSet(filename) as ds:
                for station in stations:
                    del ds.waveforms[station]
                    num_of_deleted_files += 1

        print(f"Removed {num_of_deleted_files} stations "
              f"from the LASIF project.")


    def _validate_station_and_waveform_availability(self):
        """
        Checks that all waveforms have a corresponding StationXML file
        and all stations have data.
        """

        print("Confirming that station metainformation files exist for "
              "all waveforms ", end="")

        all_good = True

        # Loop over all events.
        for event_name in self.comm.events.list():
            self._flush_point()

            filename = self.comm.waveforms.get_asdf_filename(
                event_name, data_type="raw")
            ds = pyasdf.ASDFDataSet(filename)
            station_names = ds.waveforms.list()

            for station_name in station_names:
                station = ds.waveforms[station_name]
                has_stationxml = "StationXML" in station
                has_waveforms = bool(station.get_waveform_tags())

                if has_stationxml is False and has_waveforms is False:
                    continue

                elif has_stationxml is False:
                    self._add_report(
                        f"WARNING:"
                        f"No StationXML found for station "
                        f"{station_name} "
                        f"in event {event_name} \n")
                    all_good = False
                    self.files_to_be_deleted[event_name].append(station_name)
                    continue
                elif has_waveforms is False:
                    self._add_report(
                        f"WARNING:"
                        f"No waveforms found for station {station} "
                        f"in event {event_name} \n")
                    all_good = False
                    self.files_to_be_deleted[event_name].append(station_name)
                    continue

        if all_good:
            self._print_ok_message()
        else:
            self._print_fail_message()

    def _validate_event_files(self):
        """
        Validates all event files in the currently active project.

        The following tasks are performed:
            * Validate against QuakeML 1.2 scheme.
            * Check for duplicate ids amongst all QuakeML files.
            * Make sure they contain at least one origin, magnitude and focal
              mechanism object.
            * Some simply sanity checks so that the event depth is reasonable
              and the moment tensor values as well. This is rather fragile and
              mainly intended to detect values specified in wrong units.
            * Events that are too close in time. Events that are less then one
              hour apart can in general not be used for adjoint tomography.
              This will naturally also detect duplicate events.
        """
        import itertools
        import math

        print("Validating %i event files ..." % self.comm.events.count())

        def print_warning(filename, message):
            self._add_report("WARNING: File '{event_name}' "
                             "contains {msg}.\n".format(
                                 event_name=os.path.basename(filename),
                                 msg=message))

        # Performing simple sanity checks.
        print("\tPerforming some basic sanity checks ", end="")
        all_good = True
        for event in self.comm.events.get_all_events().values():
            self.files_to_be_deleted[event["event_name"]] = []
            filename = event["filename"]
            self._flush_point()
            cat = pyasdf.ASDFDataSet(filename).events
            filename = os.path.basename(filename)
            # Check that all files contain exactly one event!
            if len(cat) != 1:
                all_good = False
                print_warning(filename, "%i events instead of only one." %
                              len(cat))
            event = cat[0]

            # Sanity checks related to the origin.
            if not event.origins:
                all_good = False
                print_warning(filename, "no origin")
                continue
            origin = event.preferred_origin() or event.origins[0]
            if (origin.depth % 100.0):
                all_good = False
                print_warning(
                    filename, "a depth of %.1f meters. This kind of accuracy "
                              "seems unrealistic. The depth in the QuakeML "
                              "file has to be specified in meters. Checking "
                              "all other QuakeML files for the correct units "
                              "might be a good idea"
                    % origin.depth)
            if (origin.depth > (800.0 * 1000.0)):
                all_good = False
                print_warning(filename, "a depth of more than 800 km. This is"
                                        " likely wrong.")

            # Sanity checks related to the magnitude.
            if not event.magnitudes:
                all_good = False
                print_warning(filename, "no magnitude")
                continue

            # Sanity checks related to the focal mechanism.
            if not event.focal_mechanisms:
                all_good = False
                print_warning(filename, "no focal mechanism")
                continue

            focmec = event.preferred_focal_mechanism() or \
                event.focal_mechanisms[0]
            if not hasattr(focmec, "moment_tensor") or \
                    not focmec.moment_tensor:
                all_good = False
                print_warning(filename, "no moment tensor")
                continue

            mt = focmec.moment_tensor
            if not hasattr(mt, "tensor") or \
                    not mt.tensor:
                all_good = False
                print_warning(filename, "no actual moment tensor")
                continue
            tensor = mt.tensor

            # Convert the moment tensor to a magnitude and see if it is
            # reasonable.
            mag_in_file = event.preferred_magnitude() or event.magnitudes[0]
            mag_in_file = mag_in_file.mag
            M_0 = 1.0 / math.sqrt(2.0) * math.sqrt(
                tensor.m_rr ** 2 + tensor.m_tt ** 2 + tensor.m_pp ** 2)
            magnitude = 2.0 / 3.0 * math.log10(M_0) - 6.0
            # Use some buffer to account for different magnitudes.
            if not (mag_in_file - 1.0) < magnitude < (mag_in_file + 1.0):
                all_good = False
                print_warning(
                    filename, "a moment tensor that would result in a moment "
                              "magnitude of %.2f. The magnitude specified in "
                              "the file is %.2f. Please check that all "
                              "components of the tensor are in Newton * meter"
                    % (magnitude, mag_in_file))

        if all_good is True:
            self._print_ok_message()
        else:
            self._print_fail_message()

        # Collect event times
        event_infos = self.comm.events.get_all_events().values()

        # Now check the time distribution of events.
        print("\tChecking for duplicates and events too close in time %s" %
              (self.comm.events.count() * "."), end="")
        all_good = True
        # Sort the events by time.
        event_infos = sorted(event_infos, key=lambda x: x["origin_time"])
        # Loop over adjacent indices.
        a, b = itertools.tee(event_infos)
        next(b, None)
        for event_1, event_2 in zip(a, b):
            time_diff = abs(event_2["origin_time"] - event_1["origin_time"])
            # If time difference is under one hour, it could be either a
            # duplicate event or interfering events.
            if time_diff <= 3600.0:
                all_good = False
                self._add_report(
                    "WARNING: "
                    "The time difference between events '{file_1}' and "
                    "'{file_2}' is only {diff:.1f} minutes. This could "
                    "be either due to a duplicate event or events that have "
                    "interfering waveforms.\n".format(
                        file_1=event_1["filename"],
                        file_2=event_2["filename"],
                        diff=time_diff / 60.0))
        if all_good is True:
            self._print_ok_message()
        else:
            self._print_fail_message()

        # Check that all events fall within the chosen boundaries.
        print("\tAssure all events are in chosen domain %s" %
              (self.comm.events.count() * "."), end="")
        all_good = True
        domain = self.comm.project.domain
        for event in event_infos:
            if domain.point_in_domain(latitude=event["latitude"],
                                      longitude=event["longitude"]):
                continue
            all_good = False
            self._add_report(
                "\nWARNING: "
                "Event '{filename}' is out of bounds of the chosen domain."
                "\n".format(filename=event["filename"]))
        if all_good is True:
            self._print_ok_message()
        else:
            self._print_fail_message()

    def is_event_station_raypath_within_boundaries(
            self, event_name, station_latitude, station_longitude,
            raypath_steps=25):
        """
        Checks if the full station-event raypath is within the project's domain
        boundaries.

        Returns True if this is the case, False if not.

        :type event_latitude: float
        :param event_latitude: The event latitude.
        :type event_longitude: float
        :param event_longitude: The event longitude.
        :type station_latitude: float
        :param station_latitude: The station latitude.
        :type station_longitude: float
        :param station_longitude: The station longitude.
        :type raypath_steps: int
        :param raypath_steps: The number of discrete points along the raypath
            that will be checked. Optional.
        """
        from lasif.utils import greatcircle_points, Point
        ev = self.comm.events.get(event_name)

        domain = self.comm.project.domain

        # Short circuit.
        if domain.is_global_domain():
            return True

        for point in greatcircle_points(
                Point(station_latitude, station_longitude),
                Point(ev["latitude"], ev["longitude"]),
                max_npts=raypath_steps):

            if not domain.point_in_domain(latitude=point.lat,
                                          longitude=point.lng):
                return False
        return True
