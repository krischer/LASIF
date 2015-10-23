#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import fnmatch
import os
import warnings

from lasif import LASIFNotFoundError, LASIFWarning
from .component import Component


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
    def __init__(self, data_folder, synthetics_folder, communicator,
                 component_name):
        self._data_folder = data_folder
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
            return os.path.join(self._data_folder, event_name, "raw.h5")
        elif data_type == "processed":
            if not tag_or_iteration:
                msg = "Tag must be given for processed data."
                raise ValueError(msg)
            return os.path.join(self._data_folder, event_name,
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
                event_name + ".h5")
        else:
            raise ValueError("Invalid data type '%s'." % data_type)

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

    def get_waveforms_synthetic(self, event_name, station_id,
                                long_iteration_name):
        """
        Gets the synthetic waveforms for the given event and station as a
        :class:`~obspy.core.stream.Stream` object.

        :param event_name: The name of the event.
        :param station_id: The id of the station in the form ``NET.STA``.
        :param long_iteration_name: The long form of an iteration name.
        """
        from lasif import rotations
        import lasif.domain

        iteration = self.comm.iterations.get(long_iteration_name)

        st = self._get_waveforms(event_name, station_id,
                                 data_type="synthetic",
                                 tag_or_iteration=iteration.long_name)
        network, station = station_id.split(".")

        # In the case of data coming from SES3D the components must be
        # mapped to ZNE as it works in XYZ.
        if "ses3d" in iteration.solver_settings["solver"].lower():
            # This maps the synthetic channels to ZNE.
            synthetic_coordinates_mapping = {"X": "N",
                                             "x": "N",
                                             "Y": "E",
                                             "y": "E",
                                             "Z": "Z",
                                             "z": "Z"}

            for tr in st:
                tr.stats.network = network
                tr.stats.station = station
                # SES3D X points south. Reverse it to arrive at ZNE.
                if tr.stats.channel in ["X", "x"]:
                    tr.data *= -1.0
                # SES3D files have no starttime. Set to the event time.
                tr.stats.starttime = \
                    self.comm.events.get(event_name)["origin_time"]
                tr.stats.channel = \
                    synthetic_coordinates_mapping[tr.stats.channel]

            # Rotate if needed. Again only SES3D synthetics need to be rotated.
            domain = self.comm.project.domain
            if isinstance(domain, lasif.domain.RectangularSphericalSection) \
                    and domain.rotation_angle_in_degree and \
                    "ses3d" in iteration.solver_settings["solver"].lower():
                # Coordinates are required for the rotation.
                coordinates = self.comm.query.get_coordinates_for_station(
                    event_name, station_id)

                # First rotate the station back to see, where it was
                # recorded.
                lat, lng = rotations.rotate_lat_lon(
                    lat=coordinates["latitude"], lon=coordinates["longitude"],
                    rotation_axis=domain.rotation_axis,
                    angle=-domain.rotation_angle_in_degree)
                # Rotate the synthetics.
                n, e, z = rotations.rotate_data(
                    st.select(channel="N")[0].data,
                    st.select(channel="E")[0].data,
                    st.select(channel="Z")[0].data,
                    lat, lng,
                    domain.rotation_axis,
                    domain.rotation_angle_in_degree)
                st.select(channel="N")[0].data = n
                st.select(channel="E")[0].data = e
                st.select(channel="Z")[0].data = z

        st.sort()

        # Apply the project function that modifies synthetics on the fly.
        fct = self.comm.project.get_project_function("process_synthetics")
        return fct(st, iteration=iteration,
                   event=self.comm.events.get(event_name))

    def _get_waveforms(self, event_name, station_id, data_type,
                       tag_or_iteration=None):
        import pyasdf

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

            return st

    def _assert_tags(self, station_group, data_type, filename):
        """
        Asserts the available tags and returns a single tag.
        """
        station_id = station_group._station_name

        tags = station_group.get_waveform_tags()
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
            assert "synthetic" in tag, (
                "The tag for station '%s' in file '%s' must contain "
                "'synthetic' for synthetic data." % (station_id, filename))
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
        import pyasdf

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
            print(filename)
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

    def get_available_processing_tags(self, event_name):
        """
        Returns the available processing tags for a given event.

        :param event_name: The event name.
        """
        data_dir = os.path.join(self._data_folder, event_name)
        if not os.path.exists(data_dir):
            raise LASIFNotFoundError("No data for event '%s'." % event_name)
        tags = []
        for tag in os.listdir(data_dir):
            # Only interested in preprocessed data.
            if not tag.startswith("preprocessed") or \
                    tag.endswith("_cache.sqlite"):
                continue
            tags.append(tag)
        return tags

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
