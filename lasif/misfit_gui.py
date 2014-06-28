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
import matplotlib
# The OSX backend has a problem with blitting. Default to TkAgg instead. If
# another backend is being used, don't do anything.
if "osx" in matplotlib.get_backend().lower():
    matplotlib.use('TkAgg')
matplotlib.rcParams['toolbar'] = 'None'

# Deactivate all default key shortcuts.
for key in matplotlib.rcParams.iterkeys():
    if not key.startswith("keymap."):
        continue
    matplotlib.rcParams[key] = ''

from obspy.core.util.geodetics import locations2degrees
from obspy.taup.taup import getTravelTimes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib.widgets import Button, MultiCursor

from matplotlib_selection_rectangle import WindowSelectionRectangle

import warnings

from lasif.window_selection import select_windows
from lasif import visualization
from lasif.adjoint_sources.ad_src_tf_phase_misfit import adsrc_tf_phase_misfit


HELP_TEXT = r"""
LASIF Misfit GUI

Usage:

* Click and drag with the left mouse button to select a window
* Click on a window with the right mouse button to delete it
* Click on a window with the left mouse button to display the misfit again

Key bindings:

* 'w', followed by a weight to set weight for all the following windows
* 'n' loads the next station
* 'p' loads the previous station
* 'r' deletes all windows for the current station
""".strip()


# Configuration for the key mappings.
KEYMAP = {
    "show help": "h",
    "set weight": "w",
    "next station": "n",
    "previous station": "p",
    "reset station": "r"}


class MisfitGUI:
    def __init__(self, event, seismogram_generator, project, window_manager,
                 adjoint_source_manager, iteration):
        # gather basic information --------------------------------------------
        plt.figure(figsize=(22, 12))
        self.event = event
        self.project = project
        self.process_parameters = iteration.get_process_params()

        self.adjoint_source_manager = adjoint_source_manager
        self.seismogram_generator = seismogram_generator
        self.window_manager = window_manager

        self._current_app_mode = None

        # setup necessary info for plot layout and buttons --------------------
        self.__setup_plots()
        self.__connect_signals()

        # read seismograms, show the gui, print basic info on the screen ------
        self.next()
        plt.gcf().canvas.set_window_title("Misfit GUI - Press 'h' for help.")
        plt.show()

    def _activate_multicursor(self):

        self._deactivate_multicursor
        self.multicursor = MultiCursor(plt.gcf().canvas, (
            self.plot_axis_z, self.plot_axis_n, self.plot_axis_e),
            color="blue", lw=1)

    def _deactivate_multicursor(self):
        try:
            self.multicursor.clear()
        except:
            pass
        try:
            del self.multicursor
        except:
            pass

    def __setup_plots(self):

        # Some actual plots. [left, bottom, width, height]
        self.plot_axis_z = plt.axes([0.10, 0.80, 0.70, 0.19])
        self.plot_axis_n = plt.axes([0.10, 0.60, 0.70, 0.19])
        self.plot_axis_e = plt.axes([0.10, 0.40, 0.70, 0.19])


        # Append another attribute to the plot axis to be able to later on identify which component they belong to.
        self.plot_axis_z.seismic_component = "Z"
        self.plot_axis_n.seismic_component = "N"
        self.plot_axis_e.seismic_component = "E"

        self._activate_multicursor()

        self.misfit_axis = plt.axes([0.10, 0.05, 0.70, 0.30])
        self.colorbar_axis = plt.axes([0.01, 0.05, 0.01, 0.30])
        
        self.map_axis = plt.axes([0.805, 0.40, 0.19, 0.19])

        # Plot the map and the beachball.
        bounds = self.project.domain["bounds"]
        self.map_obj = visualization.plot_domain(
            bounds["minimum_latitude"], bounds["maximum_latitude"],
            bounds["minimum_longitude"], bounds["maximum_longitude"],
            bounds["boundary_width_in_degree"],
            rotation_axis=self.project.domain["rotation_axis"],
            rotation_angle_in_degree=self.project.domain["rotation_angle"],
            plot_simulation_domain=False, zoom=True, labels=False)
        visualization.plot_events([self.event], map_object=self.map_obj)

        # All kinds of buttons [left, bottom, width, height]
        self.axnext = plt.axes([0.85, 0.95, 0.08, 0.03])
        self.axprev = plt.axes([0.85, 0.90, 0.08, 0.03])
        self.axreset = plt.axes([0.85, 0.85, 0.08, 0.03])
        self.axautopick = plt.axes([0.85, 0.80, 0.08, 0.03])
        self.bnext = Button(self.axnext, 'Next')
        self.bprev = Button(self.axprev, 'Prev')
        self.breset = Button(self.axreset, 'Reset Station')
        self.bautopick = Button(self.axautopick, 'Autoselect')

        # Axis displaying the current weight
        self.axweight = plt.axes([0.85, 0.75, 0.08, 0.03])
        self._update_current_weight(1.0)

    def __connect_signals(self):
        self.bnext.on_clicked(self.next)
        self.bprev.on_clicked(self.prev)
        self.breset.on_clicked(self.reset)
        self.bautopick.on_clicked(self.autoselect_windows)

        self.plot_axis_z.figure.canvas.mpl_connect('button_press_event', self._onButtonPress)
        self.plot_axis_n.figure.canvas.mpl_connect('button_press_event', self._onButtonPress)
        self.plot_axis_e.figure.canvas.mpl_connect('button_press_event', self._onButtonPress)

        self.plot_axis_e.figure.canvas.mpl_connect('key_release_event', self._onKeyRelease)

        # Connect the picker. Connecting once is enough. It will still fire for all axes.
        self.plot_axis_z.figure.canvas.mpl_connect('pick_event', self._on_pick)

    def autoselect_windows(self, event):
        """
        Automatically select proper time windows.
        """

        for component in ["Z", "N", "E"]:
            real_trace = self.data.data.select(component=component)
            synth_trace = self.data.synthetics.select(channel=component)
            if not real_trace or not synth_trace:
                continue
            real_trace = real_trace[0]
            synth_trace = synth_trace[0]

            windows = select_windows(
                real_trace, synth_trace, self.event["latitude"],
                self.event["longitude"], self.event["depth_in_km"],
                self.data.coordinates["latitude"],
                self.data.coordinates["longitude"],
                1.0 / self.process_parameters["lowpass"],
                1.0 / self.process_parameters["highpass"])

            for idx_start, idx_end in windows:
                window_start = self.time_axis[int(round(idx_start))]
                window_end = self.time_axis[int(round(idx_end))]
                self._onWindowSelected(window_start, window_end - window_start,
                                       axis=getattr(self,  "plot_axis_%s" %
                                                    component.lower()))

    def _on_pick(self, event):
        artist = event.artist

        # Use the right mouse button for deleting windows.
        if event.mouseevent.button == 3:
            x_min = artist.get_x()
            x_max = x_min + artist.get_width()

            artist.remove()
            self.delete_window(x_min=x_min, x_max=x_max,
                               component=artist.axes.seismic_component)
            plt.draw()

        if event.mouseevent.button == 1:
            self._onWindowSelected(artist.get_x(), artist.get_width(), artist.axes, plot_only=True)


    def next(self, *args):
        while True:
            try:
                data = self.seismogram_generator.next()
            except StopIteration:
                print "* MEASUREMENT PROCESS FINISHED *"
                return
            if not data:
                continue
            break
        self.data = data
        self.checks_and_infos()
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
        self.checks_and_infos()
        self.update()

    def update(self):
        self.selected_windows = {
            "Z": [],
            "N": [],
            "E": []}
        self.plot()
        for trace in self.data.data:
            windows = self.window_manager.get_windows(trace.id)
            if not windows or "windows" not in windows or \
                    not windows["windows"]:
                continue
            for window in windows["windows"]:
                self.plot_window(
                    component=windows["channel_id"][-1],
                    starttime=window["starttime"], endtime=window["endtime"],
                    window_weight=window["weight"])
        plt.draw()

    def checks_and_infos(self):
        # provide some basic screen output -----------------------------------
        d = self.data.data[0]
        print "============================================"
        print "station: " + d.stats.network + '.' + d.stats.station

        # loop through components and check if they are flipped --------------
        for comp in {"N", "E", "Z"}:
            # transcribe traces
            synth = self.data.synthetics.select(channel=comp)[0].data
            try:
                data = self.data.data.select(component=comp)[0].data
            except IndexError:
                return

            # compute correlation coefficient --------------------------------
            norm = np.sqrt(np.sum(data ** 2)) * np.sqrt(np.sum(synth ** 2))
            cc = np.sum(data * synth) / norm

            # flip traces if correlation coefficient is close to -1 ----------
            if cc < (-0.7):
                self.data.data.select(component=comp)[0].data = -data
                print "correlation coefficient below -0.7, data fliped"

        print "============================================"

    def delete_window(self, x_min, x_max, component):
        trace = self.data.data.select(component=component)[0]
        starttime = trace.stats.starttime + x_min
        endtime = trace.stats.starttime + x_max
        channel_id = trace.id
        self.window_manager.delete_window(channel_id, starttime, endtime)

    def plot_window(self, component, starttime, endtime, window_weight):
        if component == "Z":
            axis = self.plot_axis_z
        elif component == "N":
            axis = self.plot_axis_n
        elif component == "E":
            axis = self.plot_axis_e
        else:
            raise NotImplementedError

        trace = self.data.synthetics[0]

        ymin, ymax = axis.get_ylim()
        xmin = starttime - trace.stats.starttime
        width = endtime - starttime
        height = ymax - ymin
        rect = Rectangle((xmin, ymin), width, height, facecolor="0.6",
                         alpha=0.5, edgecolor="0.5", picker=True)
        axis.add_patch(rect)
        attached_text = axis.text(
            x=xmin + 0.02 * width, y=ymax - 0.02 * height,
            s=str(window_weight), verticalalignment="top",
            horizontalalignment="left", color="0.4", weight=1000)

        # Monkey patch to trigger text removal as soon as the rectangle is removed.
        def remove():
            super(Rectangle, rect).remove()
            attached_text.remove()
        rect.remove = remove

    def reset(self, *args):
        for trace in self.data.data:
            self.window_manager.delete_windows(trace.id)
        self.update()

    def plot(self):
        self.misfit_axis.cla()
        self.misfit_axis.set_yticks([])
        self.colorbar_axis.cla()
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

        s_stats = self.data.synthetics[0].stats
        self.time_axis = np.linspace(0, s_stats.npts * s_stats.delta, s_stats.npts)

        def plot_trace(axis, component):
            real_trace = self.data.data.select(component=component)
            synth_trace = self.data.synthetics.select(channel=component)
            if real_trace:
                axis.plot(self.time_axis, real_trace[0].data, color="black")
                if component != "E":
                    axis.set_xticks([])
            if synth_trace:
                axis.plot(self.time_axis, synth_trace[0].data, color="red")
                if component != "E":
                    axis.set_xticks([])

            if real_trace:
                text = real_trace[0].id
                axis.text(
                    x=0.01, y=0.95, s=text, transform=axis.transAxes,
                    bbox=dict(facecolor='white', alpha=0.5),
                    verticalalignment="top")
                axis.text(
                    x=0.01, y=0.05, s="Data Scaling Factor: %.3g" %
                    real_trace[0].stats.scaling_factor,
                    transform=axis.transAxes, bbox=dict(
                        facecolor='white', alpha=0.5),
                    verticalalignment="bottom", horizontalalignment="left")
            else:
                text = "No data, component %s" % component
                axis.text(
                    x=0.01, y=0.95, s=text, transform=axis.transAxes,
                    bbox=dict(facecolor='red', alpha=0.5),
                    verticalalignment="top")

            axis.set_xlim(self.time_axis[0], self.time_axis[-1])
            axis.set_ylabel("m/s")
            axis.grid()

        plot_trace(self.plot_axis_z, "Z")
        plot_trace(self.plot_axis_n, "N")
        plot_trace(self.plot_axis_e, "E")

        self.plot_axis_e.set_xlabel("Seconds since Event")

        self.plot_traveltimes()
        self.plot_raypath()

        plt.draw()

    def plot_traveltimes(self):
        great_circle_distance = locations2degrees(
            self.event["latitude"], self.event["longitude"],
            self.data.coordinates["latitude"],
            self.data.coordinates["longitude"])
        tts = getTravelTimes(great_circle_distance, self.event["depth_in_km"],
                             model="ak135")
        for component in ["z", "n", "e"]:
            axis = getattr(self, "plot_axis_%s" % component)
            ymin, ymax = axis.get_ylim()
            for phase in tts:
                if phase["phase_name"].lower().startswith("p"):
                    color = "green"
                else:
                    color = "red"
                axis.axvline(x=phase["time"], ymin=-1, ymax=+1,
                             color=color, alpha=0.5)
            axis.set_ylim(ymin, ymax)

    def plot_raypath(self):
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
            self.data.coordinates["longitude"],
            self.data.coordinates["latitude"],
            self.event["longitude"], self.event["latitude"], linewidth=2,
            color='green', ax=self.map_axis)

        lng, lats = self.map_obj([self.data.coordinates["longitude"]],
                                 [self.data.coordinates["latitude"]])
        self.station_icon = self.map_axis.scatter(
            lng, lats, facecolor="blue", edgecolor="black", zorder=10000,
            marker="^", s=40)

    def _onKeyRelease(self, event):
        """
        Fired on all key releases.

        It essentially is a simple state manager with the key being routed to
        different places depending on the current application state.
        """
        key = event.key

        # Default mode is no mode.
        if self._current_app_mode is None:
            # Enter help mode.
            if key == KEYMAP["show help"]:
                self._current_app_mode = "help"
                self._axes_to_restore = plt.gcf().axes
                self.help_axis = plt.gcf().add_axes((0, 0, 1, 1))
                self.help_axis.text(
                    0.5, 0.8, HELP_TEXT, transform=self.help_axis.transAxes,
                    verticalalignment="center", horizontalalignment="center")
                self.help_axis.set_xticks([])
                self.help_axis.set_yticks([])
                self._deactivate_multicursor()
                plt.draw()
            # Enter weight selection mode.
            elif key == KEYMAP["set weight"]:
                self._current_app_mode = "weight_selection"
                self._current_weight = ""
                self._update_current_weight("", editing=True)
            # Navigation
            elif key == KEYMAP["next station"]:
                self.next()
            elif key == KEYMAP["previous station"]:
                self.prev()
            elif key == KEYMAP["reset station"]:
                self.reset()
            return

        # Weight selection mode.
        elif self._current_app_mode == "weight_selection":
            weight_keys = "0123456789."
            # Keep typing the new weight.
            if key in weight_keys:
                self._current_weight += key
                self._update_current_weight(self._current_weight, editing=True)
            # Set the new weight. If that fails, reset it. In any case, leave
            # the weight selection mode.
            else:
                self._current_app_mode = None
                try:
                    weight = float(self._current_weight)
                except:
                    self._update_current_weight(self.weight)
                    return
                self._update_current_weight(weight)

        # Help selection mode. Any keypress while in it will exit it.
        elif self._current_app_mode == "help":
            plt.delaxes(self.help_axis)
            del self.help_axis
            for ax in self._axes_to_restore:
                plt.gcf().add_axes(ax)
            del self._axes_to_restore

            self._activate_multicursor()
            plt.draw()
            self._current_app_mode = None
            return

    def _update_current_weight(self, weight, editing=False):
        try:
            self._weight_text.remove()
        except:
            pass

        self._weight_text = self.axweight.text(
            x=0.5, y=0.5, s="Weight: %s" % str(weight),
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
                data = self.data.data.select(component="Z")
                if not data:
                    return
                self.rect = WindowSelectionRectangle(event, self.plot_axis_z,
                                                     self._onWindowSelected)
            if event.inaxes == self.plot_axis_n:
                data = self.data.data.select(component="N")
                if not data:
                    return
                self.rect = WindowSelectionRectangle(event, self.plot_axis_n,
                                                     self._onWindowSelected)
            if event.inaxes == self.plot_axis_e:
                data = self.data.data.select(component="E")
                if not data:
                    return
                self.rect = WindowSelectionRectangle(event, self.plot_axis_e,
                                                     self._onWindowSelected)

    def _onWindowSelected(self, window_start, window_width, axis,
                          plot_only=False):
        """
        Function called upon window selection.

        :param plot_only: If True, do not write anything to disk, but only
            plot.
        """
        #  Initialisation -----------------------------------------------------

        # Minimum window length is 50 samples.
        delta = self.data.synthetics[0].stats.delta
        if window_width < 50 * delta:
            plt.draw()
            return

        data = self.data.data.select(component=axis.seismic_component)[0]
        synth = self.data.synthetics.select(
            channel=axis.seismic_component)[0]

        if not data:
            plt.draw()
            return

        trace = data
        time_range = trace.stats.endtime - trace.stats.starttime
        plot_range = axis.get_xlim()[1] - axis.get_xlim()[0]
        starttime = trace.stats.starttime + (window_start / plot_range) * \
            time_range
        endtime = starttime + window_width / plot_range * time_range

        if plot_only is not True:
            self.window_manager.write_window(
                trace.id, starttime, endtime, self.weight, "cosine",
                "TimeFrequencyPhaseMisfitFichtner2008")
            self.plot_window(component=trace.id[-1], starttime=starttime,
                             endtime=endtime, window_weight=self.weight)

        #  window data and synthetics -----------------------------------------

        #  Decimal percentage of cosine taper (ranging from 0 to 1). Set to the
        # fraction of the minimum period to the window length.
        taper_percentage = np.min(
            [1.0, 1.0 / self. process_parameters["lowpass"] / window_width])

        #  data
        data_trimmed = data.copy()
        data_trimmed.trim(starttime, endtime)
        data_trimmed.taper(type='cosine',
                           max_percentage=0.5 * taper_percentage)
        data_trimmed.trim(synth.stats.starttime, synth.stats.endtime, pad=True,
                          fill_value=0.0)

        #  synthetics
        synth_trimmed = synth.copy()
        synth_trimmed.trim(starttime, endtime)
        synth_trimmed.taper(type='cosine',
                            max_percentage=0.5 * taper_percentage)
        synth_trimmed.trim(synth.stats.starttime, synth.stats.endtime,
                           pad=True, fill_value=0.0)

        #  make time axis
        t = np.linspace(0, synth.stats.npts * synth.stats.delta, synth.stats.npts)

        #  clear axes of misfit plot ------------------------------------------

        self.misfit_axis.cla()
        self.colorbar_axis.cla()
        try:
            self.misfit_axis.twin_axis.cla()
            self.misfit_axis.twin_axis.set_xticks([])
            self.misfit_axis.twin_axis.set_yticks([])
        except:
            pass

        #  set data and synthetics, compute actual misfit ---------------------

        t = np.require(t, dtype="float64", requirements="C")
        data_d = np.require(data_trimmed.data, dtype="float64", requirements="C")
        synth_d = np.require(synth_trimmed.data, dtype="float64", requirements="C")

        #  compute misfit and adjoint source
        adsrc = adsrc_tf_phase_misfit(
            t, data_d, synth_d,
            1.0 / self.process_parameters["lowpass"],
            1.0 / self.process_parameters["highpass"],
            axis=self.misfit_axis,
            colorbar_axis=self.colorbar_axis)

        #  plot misfit distribution -------------------------------------------

        # Format all the axis.
        self.misfit_axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        self.misfit_axis.twin_axis.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))
        self.colorbar_axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        plt.draw()

        #  write adjoint source to file ---------------------------------------

        if plot_only is not True:
            self.adjoint_source_manager.write_adjoint_src(adsrc["adjoint_source"], trace.id, starttime, endtime)
