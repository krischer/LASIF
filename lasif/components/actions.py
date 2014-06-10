#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import itertools
import os
import warnings

from lasif import LASIFWarning
from .component import Component


class ActionsComponent(Component):
    """
    Component implementing actions on the data. Requires most other
    components to be available.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def preprocess_data(self, iteration_name, event_names=None,
                        waiting_time=4.0):
        """
        Preprocesses all data for a given iteration.

        :param waiting_time: The time spent sleeping after the initial message
            has been printed. Useful if the user should be given the chance to
            cancel the processing.
        :param event_names: event_ids is a list of events to process in this run.
            It will process all events if not given.
        """
        import colorama
        from lasif import preprocessing
        import obspy

        iteration = self.comm.iterations.get(iteration_name)

        process_params = iteration.get_process_params()
        processing_tag = iteration.get_processing_tag()

        logfile = os.path.join(
            self.comm.project.get_output_folder("data_preprocessing"),
            "log.txt")

        def processing_data_generator():
            """
            Generate a dictionary with information for processing for each
            waveform.
            """
            # Loop over the chosen events.
            for event_name, event in iteration.events.iteritems():
                # None means to process all events, otherwise it will be a list
                # of events.
                if not ((event_names is None) or (event_name in event_names)):
                    continue

                output_folder = self.comm.waveforms.get_waveform_folder(
                    event_name=event_name, waveform_type="processed",
                    tag_or_long_iteration_name=processing_tag)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                # Get the event.
                event = self.comm.events.get(event_name)
                # Get the stations.
                stations = self.comm.query\
                    .get_all_stations_for_event(event_name)
                # Get the raw waveform data.
                waveforms = \
                    self.comm.waveforms.get_metadata_raw(event_name)

                # Group by station name.
                func = lambda x: ".".join(x["channel_id"].split(".")[:2])
                for station_name, channels in  \
                        itertools.groupby(waveforms, func):
                    # Filter waveforms with no available station files
                    # or coordinates.
                    if station_name not in stations:
                        continue
                    # Group by location.
                    locations = list(itertools.groupby(
                        channels, lambda x: x["channel_id"].split(".")[2]))
                    locations.sort(key=lambda x: x[0])

                    if len(locations) > 1:
                        msg = ("More than one location found for event "
                               "'%s' at station '%s'. The alphabetically "
                               "first one will be chosen." %
                               (event_name, station_name))
                        warnings.warn(msg, LASIFWarning)
                    location = locations[0][1]

                    # Loop over each found channel.
                    for channel in location:
                        channel.update(stations[station_name])
                        input_filename = channel["filename"]
                        output_filename = os.path.join(
                            output_folder,
                            os.path.basename(input_filename))
                        # Skip already processed files.
                        if os.path.exists(output_filename):
                            continue

                        ret_dict = {
                            "process_params": process_params,
                            "input_filename": input_filename,
                            "output_filename": output_filename,
                            "station_coordinates": {
                                "latitude": channel["latitude"],
                                "longitude": channel["longitude"],
                                "elevation_in_m": channel["elevation_in_m"],
                                "local_depth_in_m": channel[
                                    "local_depth_in_m"],
                            },
                            "station_filename": self.comm.stations
                                .get_channel_filename(
                                channel["channel_id"], channel["starttime"]),
                            "event_information": event,
                        }
                        yield ret_dict

        file_count = preprocessing.launch_processing(
            processing_data_generator(), log_filename=logfile,
            waiting_time=waiting_time, process_params=process_params)

        print("\nFinished processing %i files." %
              file_count["total_file_count"])
        if file_count["failed_file_count"]:
            print("\t%s%i files failed being processed.%s" %
                  (colorama.Fore.RED, file_count["failed_file_count"],
                   colorama.Fore.RESET))
        if file_count["warning_file_count"]:
            print("\t%s%i files raised warnings while being processed.%s" %
                  (colorama.Fore.YELLOW, file_count["warning_file_count"],
                   colorama.Fore.RESET))
        print("\t%s%i files have been processed without errors or warnings%s" %
              (colorama.Fore.GREEN, file_count["successful_file_count"],
               colorama.Fore.RESET))

        print("Logfile written to '%s'." % os.path.relpath(logfile))
