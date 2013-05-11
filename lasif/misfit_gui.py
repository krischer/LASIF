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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

from matplotlib_selection_rectangle import WindowSelectionRectangle

from lasif import visualization


class MisfitGUI:
    def __init__(self, event, seismogram_generator, project, window_manager):
        self.event = event
        self.event_latitude = event.origins[0].latitude
        self.event_longitude = event.origins[0].longitude
        self.project = project

        self.seismogram_generator = seismogram_generator

        self.window_manager = window_manager

        self.__setup_plots()
        self.__connect_signals()

        self.next()
        plt.tight_layout()
        plt.show()

    def __setup_plots(self):
        # Some actual plots.
        self.plot_axis_z = plt.subplot2grid((6, 8), (0, 0), colspan=7)
        self.plot_axis_n = plt.subplot2grid((6, 8), (1, 0), colspan=7)
        self.plot_axis_e = plt.subplot2grid((6, 8), (2, 0), colspan=7)

        self.misfit_axis = plt.subplot2grid((6, 8), (3, 0), colspan=4,
            rowspan=1)
        self.adjoint_source_axis = plt.subplot2grid((6, 8), (4, 0), colspan=4,
            rowspan=1)
        self.map_axis = plt.subplot2grid((6, 8), (3, 4), colspan=4, rowspan=3)

        # Plot the map and the beachball.
        bounds = self.project.domain["bounds"]
        self.map_obj = visualization.plot_domain(bounds["minimum_latitude"],
            bounds["maximum_latitude"], bounds["minimum_longitude"],
            bounds["maximum_longitude"], bounds["boundary_width_in_degree"],
            rotation_axis=self.project.domain["rotation_axis"],
            rotation_angle_in_degree=self.project.domain["rotation_angle"],
            plot_simulation_domain=False, show_plot=False, zoom=True)
        visualization.plot_events([self.event], map_object=self.map_obj)

        # All kinds of buttons [left, bottom, width, height]
        self.axnext = plt.axes([0.02, 0.02, 0.1, 0.03])
        self.axprev = plt.axes([0.14, 0.02, 0.1, 0.03])
        self.axreset = plt.axes([0.26, 0.02, 0.1, 0.03])
        self.bnext = Button(self.axnext, 'Next')
        self.bprev = Button(self.axprev, 'Prev')
        self.breset = Button(self.axreset, 'Reset')

    def __connect_signals(self):
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)
        self.breset.on_clicked(self.reset)

        self.plot_axis_z.figure.canvas.mpl_connect('button_press_event',
            self._onButtonPress)
        self.plot_axis_n.figure.canvas.mpl_connect('button_press_event',
            self._onButtonPress)
        self.plot_axis_e.figure.canvas.mpl_connect('button_press_event',
            self._onButtonPress)

        self.plot_axis_z.figure.canvas.mpl_connect('resize_event',
            self._on_resize)

    def _on_resize(self, *args):
        plt.tight_layout()

    def next(self, *args):
        data = self.seismogram_generator.next()
        if not data:
            return
        self.data = data
        self.update()

    def prev(self, *args):
        data = self.seismogram_generator.prev()
        if not data:
            return
        self.data = data
        self.update()

    def update(self):
        self.selected_windows = {
            "Z": [],
            "N": [],
            "E": []}
        self.plot()
        for trace in self.data["data"]:
            windows = self.window_manager.get_windows(trace.id)
            if not windows or "windows" not in windows or \
                    not windows["windows"]:
                continue
            for window in windows["windows"]:
                self.plot_window(component=windows["channel_id"][-1],
                    starttime=window["starttime"], endtime=window["endtime"])
        plt.draw()

    def plot_window(self, component, starttime, endtime):
        if component == "Z":
            axis = self.plot_axis_z
        elif component == "N":
            axis = self.plot_axis_n
        elif component == "E":
            axis = self.plot_axis_e
        else:
            raise NotImplementedError

        trace = self.data["synthetics"][0]

        ymin, ymax = axis.get_ylim()
        rect = Rectangle((starttime - trace.stats.starttime, ymin),
            endtime - starttime, ymax - ymin, color="0.6",
            alpha=0.5, edgecolor="0.5")
        axis.add_patch(rect)

    def reset(self, event):
        self.update()

    def plot(self, swap_polarization=False):
        self.adjoint_source_axis.cla()
        self.adjoint_source_axis.set_xticks([])
        self.adjoint_source_axis.set_yticks([])
        self.misfit_axis.cla()
        self.misfit_axis.set_xticks([])
        self.misfit_axis.set_yticks([])
        try:
            self.misfit_axis.twin_axis.cla()
            self.misfit_axis.twin_axis.set_xticks([])
            self.misfit_axis.twin_axis.set_yticks([])
        except:
            pass
        try:
            del self.rect
        except:
            pass
        self.rect = None

        # Clear all three plot axes.
        self.plot_axis_z.cla()
        self.plot_axis_n.cla()
        self.plot_axis_e.cla()

        s_stats = self.data["synthetics"][0].stats
        time_axis = np.linspace(0, s_stats.npts * s_stats.delta, s_stats.npts)

        def plot_trace(axis, component):
            real_trace = self.data["data"].select(component=component)
            synth_trace = self.data["synthetics"].select(component=component)
            if real_trace:
                axis.plot(time_axis, real_trace[0].data, color="black")
            if synth_trace:
                axis.plot(time_axis, synth_trace[0].data, color="red")

            if real_trace:
                text = real_trace[0].id
                axis.text(x=0.01, y=0.95, s=text, transform=axis.transAxes,
                    bbox=dict(facecolor='white', alpha=0.5),
                    verticalalignment="top")
            else:
                text = "No data, component %s" % component
                axis.text(x=0.01, y=0.95, s=text, transform=axis.transAxes,
                    bbox=dict(facecolor='red', alpha=0.5),
                    verticalalignment="top")

            axis.set_xlim(0, 500)
            axis.set_ylabel("m/s")
            axis.grid()

        plot_trace(self.plot_axis_z, "Z")
        plot_trace(self.plot_axis_n, "N")
        plot_trace(self.plot_axis_e, "E")

        self.plot_axis_e.set_xlabel("Seconds since Event")

        try:
            self.greatcircle[0].remove()
            self.greatcircle = None
        except:
            pass
        try:
            self.station_icon.remove()
            self.station_icon = None
        except:
            pass
        self.greatcircle = self.map_obj.drawgreatcircle(
            self.data["coordinates"]["longitude"],
            self.data["coordinates"]["latitude"],
            self.event_longitude, self.event_latitude, linewidth=2,
            color='green', ax=self.map_axis)

        lng, lats = self.map_obj([self.data["coordinates"]["longitude"]],
            [self.data["coordinates"]["latitude"]])
        self.station_icon = self.map_axis.scatter(lng, lats, color="blue",
            edgecolor="black", zorder=10000, marker="^", s=40)

        plt.draw()

    def _onButtonPress(self, event):
        if event.button != 1:  # or event.inaxes != self.plot_axis_z:
            return
        # Store the axis.
        if event.name == "button_press_event":
            if event.inaxes == self.plot_axis_z:
                data = self.data["data"].select(component="Z")
                if not data:
                    return
                self.rect = WindowSelectionRectangle(event, self.plot_axis_z,
                    self._onWindowSelected)
            if event.inaxes == self.plot_axis_n:
                data = self.data["data"].select(component="N")
                if not data:
                    return
                self.rect = WindowSelectionRectangle(event, self.plot_axis_n,
                    self._onWindowSelected)
            if event.inaxes == self.plot_axis_e:
                data = self.data["data"].select(component="E")
                if not data:
                    return
                self.rect = WindowSelectionRectangle(event, self.plot_axis_e,
                    self._onWindowSelected)

    def _onWindowSelected(self, window_start, window_width, axis):
        """
        Function called upon window selection.
        """
        if window_width <= 0:
            return

        if axis is self.plot_axis_z:
            data = self.data["data"].select(component="Z")
        elif axis is self.plot_axis_n:
            data = self.data["data"].select(component="N")
        elif axis is self.plot_axis_e:
            data = self.data["data"].select(component="E")
        else:
            return

        if not data:
            return
        trace = data[0]

        time_range = trace.stats.endtime - trace.stats.starttime
        plot_range = axis.get_xlim()[1] - axis.get_xlim()[0]
        starttime = trace.stats.starttime + (window_start / plot_range) * \
            time_range
        endtime = starttime + window_width / plot_range * time_range

        self.window_manager.write_window(trace.id, starttime, endtime, 1.0,
            "cosine", "TimeFrequencyPhaseMisfitFichtner2008")

        return

    def _write_adj_src(self):
        pass
        #adjoint_src_filename = os.path.join(self.adjoint_src_folder,
            #self.filename)

        #self.adjoint_source_axis.cla()
        #self.adjoint_source_axis.set_title("Adjoint Source")
        #self.adjoint_source_axis.plot(self.adjoint_source, color="black")
        #self.adjoint_source_axis.set_xlim(0, len(self.adjoint_source))
        #plt.draw()

        # Do some calculations.
        #rec_lat = self.synth_tr.stats.ses3d.receiver_latitude
        #rec_lng = self.synth_tr.stats.ses3d.receiver_longitude
        #rec_depth = self.synth_tr.stats.ses3d.receiver_depth_in_m
        ## Rotate back to rotated system.
        #rec_lat, rec_lng = rotations.rotate_lat_lon(rec_lat, rec_lng,
            #ROTATION_AXIS, -ROTATION_ANGLE)
        #rec_colat = rotations.lat2colat(rec_lat)

        ## Actually write the adjoint source file in SES3D specific format.
        #with open(adjoint_src_filename, "wt") as open_file:
            #open_file.write("-- adjoint source ------------------\n")
            #open_file.write("-- source coordinates (colat,lon,depth)\n")
            #open_file.write("%f %f %f\n" % (rec_lng, rec_colat, rec_depth))
            #open_file.write("-- source time function (x, y, z) --\n")
            #for x, y, z in izip(adjoint_src[1], -1.0 * adjoint_src[0],
                    #adjoint_src[2]):
                #open_file.write("%e %e %e\n" % (x, y, z))


def launch(event, seismogram_generator, project):
    MisfitGUI(event, seismogram_generator, project)
