#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import fnmatch
import os
import warnings
import pyasdf

from lasif import LASIFNotFoundError, LASIFWarning
from .component import Component


class LimitedSizeDict(collections.OrderedDict):
    """
    Based on http://stackoverflow.com/a/2437645/1657047
    """
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        collections.OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        collections.OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


class WaveformsComponent(Component):
    """
    Component managing the waveform data.

    The LASIF-HP branch can only deal with ASDF files so all waveform data
    is expected to be in this format.

    :param data_folder: The data folder in a LASIF project.
    :param synthetics_folder: The synthetics folder in a LASIF project.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, data_folder, preproc_data_folder, synthetics_folder,
                 communicator, component_name):
        self._data_folder = data_folder
        self._preproc_data_folder = preproc_data_folder
        self._synthetics_folder = synthetics_folder
        super(WaveformsComponent, self).__init__(communicator, component_name)

    def get_asdf_filename(self, event_name, data_type, tag_or_iteration=None):
        """
        Returns the filename for an ASDF waveform file.

        :param event_name: Name of the event.
        :param data_type: The type of data, one of ``"raw"``,
            ``"processed"``, ``"synthetic"``
        :param tag_or_iteration: The processing tag or iteration name if any.
        """
        if data_type == "raw":
            return os.path.join(self._data_folder, event_name + ".h5")
        elif data_type == "processed":
            if not tag_or_iteration:
                msg = "Tag must be given for processed data."
                raise ValueError(msg)
            return os.path.join(self._preproc_data_folder, event_name,
                                tag_or_iteration + ".h5")
        elif data_type == "synthetic":
            if not tag_or_iteration:
                msg = "Long iteration name must be given for synthetic data."
                raise ValueError(msg)
            if not tag_or_iteration.startswith("ITERATION_"):
                tag_or_iteration = "ITERATION_" + tag_or_iteration
            return os.path.join(
                self._synthetics_folder,
                tag_or_iteration,
                event_name,
                "receivers.h5")
        else:
            raise ValueError("Invalid data type '%s'." % data_type)

    @property
    def preprocessing_tag(self):
        """
        Gets the preprocessing tag for the lasif project, since each
         lasif project assumes a constant frequency
        this only has to be one tag.
        :return:
        """
        highpass_period = \
            self.comm.project.processing_params["highpass_period"]
        lowpass_period = \
            self.comm.project.processing_params["lowpass_period"]
        return "preprocessed_%is_to_%is" % (int(highpass_period),
                                            int(lowpass_period))

    def delete_station_from_raw(self, event_name, station_id):
        """
        Deletes all information from the raw data file for the
        given station.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form ``NET.STA``.
        """
        filename = self.get_asdf_filename(event_name, data_type="raw")

        with pyasdf.ASDFDataSet(filename, mode="a") as ds:
            del ds.waveforms[station_id]

    def get_waveforms_raw(self, event_name, station_id):
        """
        Gets the raw waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form ``NET.STA``.
        """
        return self._get_waveforms(event_name, station_id, data_type="raw")

    def get_waveforms_processed(self, event_name, station_id, tag):
        """
        Gets the processed waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form ``NET.STA``.
        :param tag: The processing tag.
        """
        return self._get_waveforms(event_name, station_id,
                                   data_type="processed", tag_or_iteration=tag)

    def get_waveforms_processed_on_the_fly(self, event_name, station_id):
        """
        Gets the processed waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form ``NET.STA``.
        :param tag: The processing tag.
        """
        st, inv = self._get_waveforms(event_name, station_id,
                                      data_type="raw", get_inventory=True)
        return self.process_data(st, inv, event_name)

    def get_waveforms_synthetic(self, event_name, station_id,
                                long_iteration_name):
        """
        Gets the synthetic waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form ``NET.STA``.
        :param long_iteration_name: The long form of an iteration name.
        """

        st = self._get_waveforms(event_name, station_id,
                                 data_type="synthetic",
                                 tag_or_iteration=long_iteration_name)

        return self.process_synthetics(st=st, event_name=event_name)

    def process_synthetics(self, st, event_name):
        # Apply the project function that modifies synthetics on the fly.
        fct = self.comm.project.get_project_function("process_synthetics")
        processing_parmams = self.comm.project.processing_params
        processing_parmams["salvus_start_time"] = \
            self.comm.project.solver_settings["start_time"]
        return fct(st, processing_parmams,
                   event=self.comm.events.get(event_name))

    def process_data(self, st, inv, event_name):
        """ This will process the data on the fly"""
        # Apply the project function that modifies synthetics on the fly.
        fct = self.comm.project.get_project_function("processing_function")

        processing_parmams = self.comm.project.processing_params
        processing_parmams["salvus_start_time"] = \
            self.comm.project.solver_settings["start_time"]
        processing_parmams["dt"] = \
            self.comm.project.solver_settings["time_increment"]
        processing_parmams["npts"] = \
            self.comm.project.solver_settings["number_of_time_steps"]
        processing_parmams["end_time"] =\
            self.comm.project.solver_settings["end_time"]

        return fct(st, inv, processing_parmams,
                   event=self.comm.events.get(event_name))

    def light_preprocess(self, event):
        """
        A way to cheat LASIF to be able to work with larger data sets.
        This should reduce size of data by downsampling and filtering.
        """
        fct = self.comm.project.get_project_function(
            "light_preprocessing_function")
        params = self.comm.project.processing_params

        freq = 1.0 / params["starting_period"]
        dt = 1.0 / freq / 3.0  # Sample each period at least three times

        light_proc_params = {}
        light_proc_params["max_freq"] = freq
        light_proc_params["dt"] = dt
        light_proc_params["event_file_name"] = \
            self.get_asdf_filename(event, "raw")

        temp = self.get_asdf_filename(event, "raw") + "temp"
        light_proc_params["temp_file"] = temp

        return fct(event, light_proc_params)

    def _get_waveforms(self, event_name, station_id, data_type,
                       tag_or_iteration=None, get_inventory=False):
        filename = self.get_asdf_filename(event_name=event_name,
                                          data_type=data_type,
                                          tag_or_iteration=tag_or_iteration)

        if not os.path.exists(filename):
            raise LASIFNotFoundError("No '%s' waveform data found for event "
                                     "'%s' and station '%s'." % (data_type,
                                                                 event_name,
                                                                 station_id))

        with pyasdf.ASDFDataSet(filename, mode="r") as ds:
            station_group = ds.waveforms[station_id]

            tag = self._assert_tags(station_group=station_group,
                                    data_type=data_type, filename=filename)

            # Get the waveform data.
            st = station_group[tag]

            # Make sure it only contains data from a single location.
            locs = sorted(set([tr.stats.location for tr in st]))
            if len(locs) != 1:
                msg = ("File '%s' contains %i location codes for station "
                       "'%s'. The alphabetically first one will be chosen."
                       % (filename, len(locs), station_id))
                warnings.warn(msg, LASIFWarning)

                st = st.filter(location=locs[0])

            if get_inventory:
                inv = station_group["StationXML"]
                return st, inv

            return st

    def _assert_tags(self, station_group, data_type, filename):
        """
        Asserts the available tags and returns a single tag.
        """
        station_id = station_group._station_name

        tags = station_group.get_waveform_tags()
        if len(tags) == 0:
            raise ValueError("Station '%s' in file '%s' contains no "
                             "tag. LASIF currently expects a "
                             "single waveform tag per station." % (
                                 station_id, filename))

        if len(tags) != 1:
            raise ValueError("Station '%s' in file '%s' contains more "
                             "than one tag. LASIF currently expects a "
                             "single waveform tag per station." % (
                                 station_id, filename))
        tag = tags[0]

        # Currently has some expectations on the used waveform tags.
        # Might change in the future.
        if data_type == "raw":
            assert tag == "raw_recording", (
                "The tag for station '%s' in file '%s' must be "
                "'raw_recording' for raw data." % (station_id, filename))
        elif data_type == "processed":
            assert "processed" in tag, (
                "The tag for station '%s' in file '%s' must contain "
                "'processed' for processed data." % (station_id, filename))
        elif data_type == "synthetic":
            assert "displacement" in tag, (
                "The tag for station '%s' in file '%s' must contain "
                "'displacement' for displacement data." % (station_id,
                                                           filename))
        else:
            raise ValueError

        return tags[0]

    def get_available_data(self, event_name, station_id):
        """
        Returns a dictionary with information about the available data.

        Information is specific for a given event and station.

        :param event_name: The event name.
        :param station_id: The station id.
        """
        information = {
            "raw": {},
            "processed": {},
            "synthetic": {}
        }

        # Check the raw data components.
        raw_filename = self.get_asdf_filename(event_name=event_name,
                                              data_type="raw")
        with pyasdf.ASDFDataSet(raw_filename, mode="r") as ds:
            station_group = ds.waveforms[station_id]

            tag = self._assert_tags(station_group=station_group,
                                    data_type="raw", filename=raw_filename)

            raw_components = [_i.split("__")[0].split(".")[-1][-1] for _i in
                              station_group.list() if _i.endswith("__" + tag)]
        information["raw"]["raw"] = raw_components

        # Now figure out which processing tags are available.
        folder = os.path.join(self._data_folder, event_name)
        processing_tags = [os.path.splitext(_i)[0] for _i in os.listdir(folder)
                           if os.path.isfile(os.path.join(folder, _i)) and
                           _i.endswith(".h5") and _i != "raw.h5"]

        for proc_tag in processing_tags:
            filename = self.get_asdf_filename(
                event_name=event_name, data_type="processed",
                tag_or_iteration=proc_tag)
            with pyasdf.ASDFDataSet(filename, mode="r") as ds:
                station_group = ds.waveforms[station_id]

                tag = self._assert_tags(station_group=station_group,
                                        data_type="processed",
                                        filename=filename)

                components = [_i.split("__")[0].split(".")[-1][-1] for _i in
                              station_group.list() if _i.endswith("__" + tag)]
            information["processed"][proc_tag] = components

        # And the synthetics.
        iterations = [_i.lstrip("ITERATION_") for _i in
                      os.listdir(self._synthetics_folder)
                      if _i.startswith("ITERATION_")
                      if os.path.isdir(os.path.join(self._synthetics_folder,
                                                    _i))]

        synthetic_coordinates_mapping = {"X": "N",
                                         "Y": "E",
                                         "Z": "Z",
                                         "N": "N",
                                         "E": "E"}

        for iteration in iterations:
            filename = self.get_asdf_filename(
                event_name=event_name, data_type="synthetic",
                tag_or_iteration=iteration)
            if not os.path.exists(filename):
                continue
            with pyasdf.ASDFDataSet(filename, mode="r") as ds:
                station_group = ds.waveforms[station_id]

                tag = self._assert_tags(station_group=station_group,
                                        data_type="synthetic",
                                        filename=filename)

                components = [_i.split("__")[0].split(".")[-1][-1].upper()
                              for _i in
                              station_group.list() if _i.endswith("__" + tag)]
            information["synthetic"][iteration] = [
                synthetic_coordinates_mapping[_i] for _i in components]
        return information

    def get_available_synthetics(self, event_name):
        """
        Returns the available synthetics for a given event.

        :param event_name: The event name.
        """
        data_dir = os.path.join(self._synthetics_folder, event_name)
        if not os.path.exists(data_dir):
            raise LASIFNotFoundError("No synthetic data for event '%s'." %
                                     event_name)
        iterations = []
        for folder in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(
                    self._synthetics_folder, event_name, folder)) \
                    or not fnmatch.fnmatch(folder, "ITERATION_*"):
                continue
            iterations.append(folder)

        # Make sure the iterations also contain the event and the stations.
        its = []
        for iteration in iterations:
            try:
                it = self.comm.iterations.get(iteration)
            except LASIFNotFoundError:
                continue
            if event_name not in it.events:
                continue
            its.append(it.name)
        return its
