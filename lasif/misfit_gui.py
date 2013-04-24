#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple matplotlib based utility function to help in picking windows and
calculating misfits.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
# The OSX backend has a problem with blitting.
import matplotlib
matplotlib.use('TkAgg')

from glob import iglob, glob
from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.basemap import Basemap
from obspy import read, Stream
from obspy.imaging.beachball import Beach
import os
import sys

from event_list_reader import read_event_list
from matplotlib_selection_rectangle import WindowSelectionRectangle
from misfits import l2NormMisfit
import rotations
from ses3d_file_reader import readSES3DFile
from window import Window

# Give the directories here.
REAL_DATA = "../DATA/2001.17s/"
SYNTHETIC_DATA = "../SYNTH/2001.17s/"
EVENT_LIST_FILE = "../event_list"
EVENT_INDEX = 2001

ITERATION = 52

MISFIT_WINDOW_STORAGE_DIRECTORY = "../OUTPUT/MISFIT_WINDOWS/"
ADJOINT_SOURCE_STORAGE_DIRECTORY = "../OUTPUT/ADJOINT_SOURCES/"

ROTATION_AXIS = [0.0, 1.0, 0.0]
ROTATION_ANGLE = -57.5


stations = []
# Build a list of dicts that matches real and synthetic files.
for datafile in iglob(os.path.join(REAL_DATA, "*.SAC")):
    station = os.path.basename(datafile).split(".")[0]
    stations.append(station)
stations = list(set(stations))
station_files = []
for station in stations:
    real_files = glob(os.path.join(REAL_DATA, station + ".*.SAC"))
    synthetic_files = glob(os.path.join(SYNTHETIC_DATA, station + "_[x,y,z]"))
    station_files.append({"real_files": real_files,
        "synthetic_files": synthetic_files,
        "station_name": station})


def seismogram_generator(event):
    for station in station_files:
        # Sort so they will be in x, y, z order
        synthetic_files = sorted(station["synthetic_files"])
        if not synthetic_files:
            continue
        synth_st = Stream()
        for s_file in synthetic_files:
            synth_st += readSES3DFile(s_file)
        # Sort. Now it will have E, N, and Z components in this order.
        synth_st.sort()
        # Rotate synthetics.
        r_lon = synth_st[0].stats.ses3d.receiver_longitude
        r_lat = synth_st[0].stats.ses3d.receiver_latitude
        s_lon = synth_st[0].stats.ses3d.source_longitude
        s_lat = synth_st[0].stats.ses3d.source_latitude
        n, e, z = rotations.rotate_data(synth_st[1].data, synth_st[0].data,
                synth_st[2].data, r_lat, r_lon, ROTATION_AXIS, ROTATION_ANGLE)
        synth_st[0].data = e
        synth_st[1].data = n
        synth_st[2].data = z
        # Now also rotate the source and receiver coordinates to be in true
        # coordinates.
        r_lat, r_lon = rotations.rotate_lat_lon(r_lat, r_lon, ROTATION_AXIS,
            ROTATION_ANGLE)
        s_lat, s_lon = rotations.rotate_lat_lon(s_lat, s_lon, ROTATION_AXIS,
            ROTATION_ANGLE)
        for tr in synth_st:
            tr.stats.ses3d.receiver_longitude = r_lon
            tr.stats.ses3d.receiver_latitude = r_lat
            tr.stats.ses3d.source_longitude = s_lon
            tr.stats.ses3d.source_latitude = s_lat
            # Set the correct starttime for the trace.
            tr.stats.starttime = event["time"]

        # Now attempt to find the corresponding data stream
        for synth_tr in synth_st:
            # XXX: Try to find something more robust.
            data_file = glob(os.path.join(REAL_DATA, station["station_name"] +
                ".??" + synth_tr.stats.channel + ".SAC"))
            if not data_file:
                continue

            data_tr = read(data_file[0])[0]
            # Attempt to cut it as close to the synthetic trace as possible.
            data_tr.trim(starttime=event["time"], endtime=event["time"] +
                (synth_tr.stats.endtime - synth_tr.stats.starttime), pad=True,
                fill_value=0.0)

            # Scale the data so it matches the synthetics.
            scaling_factor = synth_tr.data.ptp() / data_tr.data.ptp()
            data_tr.data *= scaling_factor

            # Convert both to the same dtype.
            data_tr.data = np.require(data_tr.data, dtype="float32")
            synth_tr.data = np.require(synth_tr.data, dtype="float32")

            # Resample the data trace to the exact sampling rate of the
            # synthetics.
            data_tr.resample(synth_tr.stats.sampling_rate)

            # Slightly convoluted way to make sure both have the exact same
            # starttime, endtime, and number of samples.
            # XXX: Possibly replace with true synchronization function.
            data_tr.trim(synth_tr.stats.starttime, synth_tr.stats.endtime,
                pad=True, fill_value=0.0)
            data_tr.stats.starttime = synth_tr.stats.starttime
            data_tr.trim(synth_tr.stats.starttime, synth_tr.stats.endtime,
                pad=True, fill_value=0.0)

            data_tr.stats.channel = data_tr.stats.channel[-1]

            yield {
                "data_trace": data_tr,
                "synth_trace": synth_tr,
                "channel_id": data_tr.id,
                "scaling_factor": scaling_factor}

# Plot setup.
plot_axis = plt.subplot2grid((4, 4), (0, 0), colspan=4)
misfit_axis = plt.subplot2grid((4, 4), (1, 2), colspan=2, rowspan=1)
adjoint_source_axis = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=1)
map_axis = plt.subplot2grid((4, 4), (1, 0), colspan=2, rowspan=2)


class Index:
    def __init__(self):
        self.greatcircle = None
        self.index = -1

        events = read_event_list(EVENT_LIST_FILE)
        self.event = events[EVENT_INDEX]
        self.event["event_index"] = EVENT_INDEX

        files = seismogram_generator(self.event)
        self.data = []
        for i, stuff in enumerate(files):
            sys.stdout.write(".")
            sys.stdout.flush()
            if i > 10:
                break
            self.data.append(stuff)
        print ""
        self._setupMap()

        self.current_data = {}

    def _setupMap(self):
        r_lngs = []
        r_lats = []
        s_lngs = []
        s_lats = []
        # Get the extension of all data.
        for data in self.data:
            synth_trace = data["synth_trace"]
            r_lngs.append(synth_trace.stats.ses3d.receiver_longitude)
            r_lats.append(synth_trace.stats.ses3d.receiver_latitude)

            s_lngs.append(synth_trace.stats.ses3d.source_longitude)
            s_lats.append(synth_trace.stats.ses3d.source_latitude)

        lngs = r_lngs + s_lngs
        lats = r_lats + s_lats
        lng_range = max(lngs) - min(lngs)
        lat_range = max(lats) - min(lats)
        buffer = 0.2
        # Setup the map view.
        # longitude of lower left hand corner of the desired map domain
        # (degrees).
        llcrnrlon = min(lngs) - buffer * lng_range
        # longitude of upper right hand corner of the desired map domain
        # (degrees).
        urcrnrlon = max(lngs) + buffer * lng_range
        # latitude of lower left hand corner of the desired map domain
        # (degrees).
        llcrnrlat = min(lats) - buffer * lat_range
        # latitude of upper right hand corner of the desired map domain
        # (degrees).
        urcrnrlat = max(lats) + buffer * lat_range
        # center of desired map domain (in degrees).
        lon_0 = urcrnrlon + ((urcrnrlon - llcrnrlon) / 2.0)
        # center of desired map domain (in degrees).
        lat_0 = urcrnrlat + ((urcrnrlat - urcrnrlon) / 2.0)

        self.map_obj = Basemap(projection="merc", llcrnrlon=llcrnrlon,
            llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
            lat_0=lat_0, lon_0=lon_0, resolution="l")

        self.map_obj.drawcoastlines()
        self.map_obj.fillcontinents()

        r_lngs, r_lats = self.map_obj(r_lngs, r_lats)
        self.map_obj.scatter(r_lngs, r_lats, color="red", zorder=10000,
            marker="^", s=40)

        # Add beachball plot.
        x, y = self.map_obj(s_lngs[0], s_lats[0])
        focmec = [self.event["Mrr"], self.event["Mtt"], self.event["Mpp"],
                self.event["Mrt"], self.event["Mrp"], self.event["Mtp"]]
        # Attempt to calculate the best beachball size.
        width = max((self.map_obj.xmax - self.map_obj.xmin,
            self.map_obj.ymax - self.map_obj.ymin)) * 0.075
        b = Beach(focmec, xy=(x, y), width=width, linewidth=1)
        b.set_zorder(200000000)
        map_axis.add_collection(b)

    def next(self, event):
        self.index += 1
        if self.index >= len(self.data):
            self.index = len(self.data) - 1
            return
        self.plot(self.index)

    def prev(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = 0
            return
        self.plot(self.index)

    def swap_polarization(self, event):
        self.plot(self.index, swap_polarization=True)

    def reset(self, event):
        self.plot(self.index)

    def plot(self, index, swap_polarization=False):
        adjoint_source_axis.cla()
        adjoint_source_axis.set_xticks([])
        adjoint_source_axis.set_yticks([])
        misfit_axis.cla()
        misfit_axis.set_xticks([])
        misfit_axis.set_yticks([])
        try:
            misfit_axis.twin_axis.cla()
            misfit_axis.twin_axis.set_xticks([])
            misfit_axis.twin_axis.set_yticks([])
        except:
            pass
        try:
            del self.rect
        except:
            pass
        self.rect = None
        self.current_data = self.data[index]
        real_trace = self.current_data["data_trace"]
        synth_trace = self.current_data["synth_trace"]
        channel_id = self.current_data["channel_id"]
        scaling_factor = self.current_data["scaling_factor"]

        plot_axis.cla()

        time_axis = np.linspace(0, synth_trace.stats.npts *
            synth_trace.stats.delta, synth_trace.stats.npts)

        if swap_polarization is True:
            real_trace.data *= -1.0
        plot_axis.plot(time_axis, real_trace.data, color="black")
        plot_axis.plot(time_axis, synth_trace.data, color="red")

        plot_axis.set_xlim(0, 500)
        plot_axis.set_title(channel_id + " -- scaling factor: " +
                str(scaling_factor))
        plot_axis.set_xlabel("Seconds since Event")
        plot_axis.set_ylabel("m/s")

        r_lon = synth_trace.stats.ses3d.receiver_longitude
        r_lat = synth_trace.stats.ses3d.receiver_latitude
        s_lon = synth_trace.stats.ses3d.source_longitude
        s_lat = synth_trace.stats.ses3d.source_latitude
        if self.greatcircle:
            self.greatcircle[0].remove()
            self.greatcircle = None
        self.greatcircle = self.map_obj.drawgreatcircle(r_lon, r_lat, s_lon,
            s_lat, linewidth=2, color='green', ax=map_axis)
        plt.draw()

    def _onButtonPress(self, event):
        if event.button != 1 or event.inaxes != plot_axis:
            return
        # Store the axis.
        if event.name == "button_press_event":
            self.rect = WindowSelectionRectangle(event, plot_axis,
                self._onWindowSelected)

    def _onWindowSelected(self, window_start, window_width):
        """
        Function called upon window selection.
        """
        if window_width <= 0:
            return
        real_trace = self.current_data["data_trace"]
        synth_trace = self.current_data["synth_trace"]

        path = SYNTHETIC_DATA
        if path.endswith("/"):
            path = path[:-1]
        extra_id = os.path.basename(path).split(".")[-1]
        additional_identifier = "iteration_%i.%s" % (ITERATION, extra_id)

        win = Window(self.event["event_index"], additional_identifier,
            self.event["time"], window_start, window_width, "cosine",
            window_options={"percentage": 0.1}, channel_id=real_trace.id)
        real_tr = real_trace.copy()
        synth_tr = synth_trace.copy()
        win.apply(real_tr)
        win.apply(synth_tr)

        misfit, adjoint_src = l2NormMisfit(real_tr.data, synth_tr.data,
            synth_tr.stats.channel, axis=misfit_axis)
        win.set_misfit("L2NormMisfit", misfit)
        win.write(MISFIT_WINDOW_STORAGE_DIRECTORY)

        if synth_tr.stats.channel == "N":
            adjoint_source = adjoint_src[0]
        elif synth_tr.stats.channel == "E":
            adjoint_source = adjoint_src[1]
        elif synth_tr.stats.channel == "Z":
            adjoint_source = adjoint_src[2]
        else:
            raise NotImplementedError

        # Assemble the adjoint source path.
        window_path = win.output_filename
        filename = os.path.basename(window_path)
        filename = os.path.splitext(filename)[0] + os.path.extsep + "adj_src"
        subfolder = os.path.basename(os.path.split(window_path)[0])
        # Make sure the folder exists.
        adjoint_src_folder = os.path.join(ADJOINT_SOURCE_STORAGE_DIRECTORY,
            subfolder)
        if not os.path.exists(adjoint_src_folder):
            os.makedirs(adjoint_src_folder)
        adjoint_src_filename = os.path.join(adjoint_src_folder, filename)

        adjoint_source_axis.cla()
        adjoint_source_axis.set_title("Adjoint Source")
        adjoint_source_axis.plot(adjoint_source, color="black")
        adjoint_source_axis.set_xlim(0, len(adjoint_source))
        plt.draw()

        # Do some calculations.
        rec_lat = synth_tr.stats.ses3d.receiver_latitude
        rec_lng = synth_tr.stats.ses3d.receiver_longitude
        rec_depth = synth_tr.stats.ses3d.receiver_depth_in_m
        # Rotate back to rotated system.
        rec_lat, rec_lng = rotations.rotate_lat_lon(rec_lat, rec_lng,
            ROTATION_AXIS, -ROTATION_ANGLE)
        rec_colat = rotations.lat2colat(rec_lat)

        # Actually write the adjoint source file in SES3D specific format.
        with open(adjoint_src_filename, "wt") as open_file:
            open_file.write("-- adjoint source ------------------\n")
            open_file.write("-- source coordinates (colat,lon,depth)\n")
            open_file.write("%f %f %f\n" % (rec_lng, rec_colat, rec_depth))
            open_file.write("-- source time function (x, y, z) --\n")
            for x, y, z in izip(adjoint_src[1], -1.0 * adjoint_src[0],
                    adjoint_src[2]):
                open_file.write("%e %e %e\n" % (x, y, z))

callback = Index()
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axswap = plt.axes([0.7, 0.15, 0.1, 0.075])
axreset = plt.axes([0.81, 0.15, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)
bswap = Button(axswap, 'Swap Pol')
bswap.on_clicked(callback.swap_polarization)
breset = Button(axreset, 'Reset')
breset.on_clicked(callback.reset)


plot_axis.figure.canvas.mpl_connect('button_press_event',
        callback._onButtonPress)

plt.tight_layout()


plt.show()
