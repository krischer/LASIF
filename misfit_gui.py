import matplotlib
matplotlib.use('TkAgg')

from glob import iglob, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.basemap import Basemap
from obspy import read, Stream
from obspy.imaging.beachball import Beach
import os
import sys

from matplotlib_selection_rectangle import WindowSelectionRectangle
from event_list_reader import read_event_list
import rotations
from ses3d_file_reader import readSES3DFile

# Give the directories here.
REAL_DATA = "../DATA/2001.17s/"
SYNTHETIC_DATA = "../SYNTH/2001.17s/"
EVENT_LIST_FILE = "../event_list"
EVENT_INDEX = 2001

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

        # Now attempt to find the corresponding data stream
        for synth_tr in synth_st:
            data_file = glob(os.path.join(REAL_DATA, station["station_name"] +
                ".??" + synth_tr.stats.channel + ".SAC"))
            if not data_file:
                continue
            data_tr = read(data_file[0])[0]
            data_tr.trim(starttime=event["time"],
                endtime=event["time"] +
                    (synth_tr.stats.endtime - synth_tr.stats.starttime),
                pad=True, fill_value=0.0)

            scaling_factor = synth_tr.data.ptp() / data_tr.data.ptp()

            data_tr.data *= scaling_factor

            data_tr.data = np.require(data_tr.data, dtype="float32")
            synth_tr.data = np.require(synth_tr.data, dtype="float32")
            data_tr.resample(synth_tr.stats.sampling_rate)
            data_tr.stats.starttime = synth_tr.stats.starttime
            data_tr.trim(endtime=synth_tr.stats.endtime, pad=True,
                fill_value=0.0)

            yield (data_tr, synth_tr,
                station["station_name"] + "." + synth_tr.stats.channel,
                scaling_factor)


# Plot setup.
plot_axis = plt.subplot2grid((3, 3), (0, 0), colspan=3)
map_axis = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)


class Index:
    def __init__(self):
        self.greatcircle = None
        self.index = -1

        events = read_event_list(EVENT_LIST_FILE)
        self.event = events[2001]

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

        self.windows = []

    def _setupMap(self):
        r_lngs = []
        r_lats = []
        s_lngs = []
        s_lats = []
        # Get the extension of all data.
        for _, synth_trace, _, _ in self.data:
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
        self.plot(self.index)

    def prev(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = 0
        self.plot(self.index)

    def swap_polarization(self, event):
        self.plot(self.index, swap_polarization=True)

    def plot(self, index, swap_polarization=False):
        real_trace, synth_trace, channel_name, scaling_factor = \
            self.data[index]
        plot_axis.cla()

        time_axis = np.linspace(0, synth_trace.stats.npts *
            synth_trace.stats.delta, synth_trace.stats.npts)

        if swap_polarization is True:
            real_trace.data *= -1.0
        plot_axis.plot(time_axis, real_trace.data, color="black")
        plot_axis.plot(time_axis, synth_trace.data, color="red")

        plot_axis.set_xlim(0, 500)
        plot_axis.set_title(channel_name + " -- scaling factor: " +
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
        if event.button != 1:
            return
        # Store the axis.
        if event.name == "button_press_event":
            self.rect = WindowSelectionRectangle(event, plot_axis)

callback = Index()
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axswap = plt.axes([0.7, 0.15, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)
bswap = Button(axswap, 'Swap Pol')
bswap.on_clicked(callback.swap_polarization)


plot_axis.figure.canvas.mpl_connect('button_press_event',
        callback._onButtonPress)


plt.show()
