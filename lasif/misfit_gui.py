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
matplotlib.rcParams['toolbar'] = 'None'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import Button, MultiCursor

from matplotlib_selection_rectangle import WindowSelectionRectangle

import warnings

from lasif import visualization
from lasif.adjoint_sources.ad_src_tf_phase_misfit import adsrc_tf_phase_misfit


class MisfitGUI:
    def __init__(self, event, seismogram_generator, project, window_manager,
            adjoint_source_manager):
        plt.figure(figsize=(22, 12))
        self.event = event
        self.event_latitude = event.origins[0].latitude
        self.event_longitude = event.origins[0].longitude
        self.project = project

        self.adjoint_source_manager = adjoint_source_manager
        self.seismogram_generator = seismogram_generator
        self.window_manager = window_manager

        self._in_weight_selection = False

        self.__setup_plots()
        self.__connect_signals()

        self.next()
        plt.tight_layout()
        plt.show()

    def __setup_plots(self):
        # Some actual plots.
        self.plot_axis_z = plt.subplot2grid((6, 20), (0, 0), colspan=18)
        self.plot_axis_n = plt.subplot2grid((6, 20), (1, 0), colspan=18)
        self.plot_axis_e = plt.subplot2grid((6, 20), (2, 0), colspan=18)

        self.multicursor = MultiCursor(plt.gcf().canvas, (self.plot_axis_z,
            self.plot_axis_n, self.plot_axis_e), color="green", lw=1)

        self.misfit_axis = plt.subplot2grid((6, 20), (3, 0), colspan=11,
            rowspan=3)
        self.colorbar_axis = plt.subplot2grid((6, 20), (3, 12), colspan=1,
            rowspan=3)
        #self.adjoint_source_axis = plt.subplot2grid((6, 8), (4, 0), colspan=4,
            #rowspan=1)
        self.map_axis = plt.subplot2grid((6, 20), (3, 14), colspan=7,
            rowspan=3)

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
        self.axnext = plt.axes([0.90, 0.95, 0.08, 0.03])
        self.axprev = plt.axes([0.90, 0.90, 0.08, 0.03])
        self.axreset = plt.axes([0.90, 0.85, 0.08, 0.03])
        self.bnext = Button(self.axnext, 'Next')
        self.bprev = Button(self.axprev, 'Prev')
        self.breset = Button(self.axreset, 'Reset Station')

        # Axis displaying the current weight
        self.axweight = plt.axes([0.90, 0.80, 0.08, 0.03])
        self._update_current_weight(1.0)

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

        self.plot_axis_e.figure.canvas.mpl_connect('key_release_event',
            self._onKeyRelease)

        self.plot_axis_z.figure.canvas.mpl_connect('resize_event',
            self._on_resize)

    def _on_resize(self, *args):
        plt.tight_layout()

    def next(self, *args):
        while True:
            try:
                data = self.seismogram_generator.next()
            except StopIteration:
                return
            if not data:
                continue
            break
        self.data = data
        self.update()

    def prev(self, *args):
        while True:
            try:
                data = self.seismogram_generator.prev()
            except StopIteration:
                return
            if not data:
                continue
            break
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
                    starttime=window["starttime"], endtime=window["endtime"],
                    window_weight=window["weight"])
        plt.draw()

    def plot_window(self, component, starttime, endtime, window_weight):
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
        xmin = starttime - trace.stats.starttime
        width = endtime - starttime
        height = ymax - ymin
        rect = Rectangle((xmin, ymin), width, height, facecolor="0.6",
            alpha=0.5, edgecolor="0.5")
        axis.add_patch(rect)
        axis.text(x=xmin + 0.02 * width, y=ymax - 0.02 * height,
            s=str(window_weight), verticalalignment="top",
            horizontalalignment="left", color="0.4", weight=1000)

    def reset(self, event):
        for trace in self.data["data"]:
            self.window_manager.delete_windows(trace.id)
        self.update()

    def plot(self):
        self.misfit_axis.cla()
        self.misfit_axis.set_xticks([])
        self.misfit_axis.set_yticks([])
        self.colorbar_axis.cla()
        self.colorbar_axis.set_xticks([])
        self.colorbar_axis.set_yticks([])
        try:
            self.misfit_axis.twin_axis.cla()
            self.misfit_axis.twin_axis.set_xticks([])
            self.misfit_axis.twin_axis.set_yticks([])
            del self.misfit_axis.twin_axis
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

        # Set a custom tick formatter.
        self.plot_axis_z.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        self.plot_axis_n.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        self.plot_axis_e.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))

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
                axis.text(x=0.01, y=0.05, s="Data Scaling Factor: %.3g" %
                    real_trace[0].stats.scaling_factor,
                    transform=axis.transAxes, bbox=dict(facecolor='white',
                        alpha=0.5), verticalalignment="bottom",
                    horizontalalignment="left")
            else:
                text = "No data, component %s" % component
                axis.text(x=0.01, y=0.95, s=text, transform=axis.transAxes,
                    bbox=dict(facecolor='red', alpha=0.5),
                    verticalalignment="top")

            axis.set_xlim(time_axis[0], time_axis[-1])
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
        self.station_icon = self.map_axis.scatter(lng, lats, facecolor="blue",
            edgecolor="black", zorder=10000, marker="^", s=40)

        plt.draw()

    def _onKeyRelease(self, event):
        start_weight_keys = "w"
        weight_keys = "0123456789."

        if self._in_weight_selection is False \
                and event.key in start_weight_keys:
            self._in_weight_selection = True
            self._current_weight = ""
            self._update_current_weight("", editing=True)
            return
        elif self._in_weight_selection is True \
                and event.key in weight_keys:
            self._current_weight += event.key
            self._update_current_weight(self._current_weight, editing=True)
        elif self._in_weight_selection is True:
            self._in_weight_selection = False
            try:
                weight = float(self._current_weight)
            except:
                self._update_current_weight(self.weight)
                return
            self._update_current_weight(weight)

    def _update_current_weight(self, weight, editing=False):
        try:
            self._weight_text.remove()
        except:
            pass

        self._weight_text = self.axweight.text(x=0.5, y=0.5,
            s="Weight: %s" % str(weight),
            transform=self.axweight.transAxes, verticalalignment="center",
            horizontalalignment="center")

        self.axweight.set_xticks([])
        self.axweight.set_yticks([])

        if editing:
            self.axweight.set_axis_bgcolor("red")
        else:
            if isinstance(weight, float):
                self.weight = weight
            else:
                msg = "Weight '%s' is not a proper float." % str(weight)
                warnings.warn(msg)
                self._update_current_weight(self.weight)
            self.axweight.set_axis_bgcolor("white")

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
            data = self.data["data"].select(component="Z")[0]
            synth = self.data["synthetics"].select(component="Z")[0]
        elif axis is self.plot_axis_n:
            data = self.data["data"].select(component="N")[0]
            synth = self.data["synthetics"].select(component="N")[0]
        elif axis is self.plot_axis_e:
            data = self.data["data"].select(component="E")[0]
            synth = self.data["synthetics"].select(component="E")[0]
        else:
            return

        if not data:
            return

        trace = data
        time_range = trace.stats.endtime - trace.stats.starttime
        plot_range = axis.get_xlim()[1] - axis.get_xlim()[0]
        starttime = trace.stats.starttime + (window_start / plot_range) * \
            time_range
        endtime = starttime + window_width / plot_range * time_range

        self.window_manager.write_window(trace.id, starttime, endtime,
            self.weight, "cosine", "TimeFrequencyPhaseMisfitFichtner2008")
        self.plot_window(component=trace.id[-1], starttime=starttime,
            endtime=endtime, window_weight=self.weight)

        # Window the data.
        data_trimmed = data.copy()
        data_trimmed.trim(starttime, endtime)
        data_trimmed.taper()
        data_trimmed.trim(synth.stats.starttime, synth.stats.endtime, pad=True,
            fill_value=0.0)
        synth_trimmed = synth.copy()
        synth_trimmed.trim(starttime, endtime)
        synth_trimmed.taper()
        synth_trimmed.trim(synth.stats.starttime, synth.stats.endtime,
            pad=True, fill_value=0.0)

        t = np.linspace(0, synth.stats.npts * synth.stats.delta,
            synth.stats.npts)

        self.misfit_axis.cla()
        self.colorbar_axis.cla()
        try:
            self.misfit_axis.twin_axis.cla()
            self.misfit_axis.twin_axis.set_xticks([])
            self.misfit_axis.twin_axis.set_yticks([])
        except:
            pass

        t = np.require(t, dtype="float64", requirements="C")
        data_d = np.require(data_trimmed.data, dtype="float64",
            requirements="C")
        synth_d = np.require(synth_trimmed.data, dtype="float64",
            requirements="C")

        adsrc = adsrc_tf_phase_misfit(t, data_d, synth_d, 5.0, 50.0,
            0.00000001, axis=self.misfit_axis,
            colorbar_axis=self.colorbar_axis)

        # Format all the axis.
        self.misfit_axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        self.misfit_axis.twin_axis.yaxis.set_major_formatter(
            FormatStrFormatter("%.2g"))
        self.colorbar_axis.yaxis.set_major_formatter(
            FormatStrFormatter("%.1f"))

        plt.tight_layout()
        plt.draw()

        self.adjoint_source_manager.write_adjoint_src(adsrc["adjoint_source"],
            trace.id, starttime, endtime)

    def _write_adj_src(self, adj_src, channel_id, starttime, endtime):
        filename = "adjoint_source_%s_%s_%s.npy" % (channel_id, str(starttime),
            str(endtime))
        filename = os.path.join(self.adjoint_source_dir, filename)

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
