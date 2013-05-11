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

from itertools import izip
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import os

from matplotlib_selection_rectangle import WindowSelectionRectangle
#from misfits import l2NormMisfit
from window import Window


from lasif import visualization

MISFIT_WINDOW_STORAGE_DIRECTORY = ("/Users/lion/Dropbox/LASIF/TurkeyExample/"
    "ADJOINT_SOURCES_AND_WINDOWS/WINDOWS")
ADJOINT_SOURCE_STORAGE_DIRECTORY = ("/Users/lion/Dropbox/LASIF/TurkeyExample/"
    "ADJOINT_SOURCES_AND_WINDOWS/ADJOINT_SOURCES")


class MisfitGUI:
    def __init__(self, event, seismogram_generator, project):
        self.event = event
        self.event_latitude = event.origins[0].latitude
        self.event_longitude = event.origins[0].longitude
        self.project = project

        self.seismogram_generator = seismogram_generator

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
        self.axreset = plt.axes([0.14, 0.02, 0.1, 0.03])
        self.bnext = Button(self.axnext, 'Next')
        self.breset = Button(self.axreset, 'Reset')

    def __connect_signals(self):
        self.bnext.on_clicked(self.next)
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
        self.data = self.seismogram_generator.next()
        self.plot()

    def reset(self, event):
        self.plot()

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
        self.greatcircle = self.map_obj.drawgreatcircle(
            self.data["coordinates"]["longitude"],
            self.data["coordinates"]["latitude"],
            self.event_longitude, self.event_latitude, linewidth=2,
            color='green', ax=self.map_axis)
        plt.draw()

    def _onButtonPress(self, event):
        if event.button != 1 or event.inaxes != self.plot_axis_z:
            return
        # Store the axis.
        if event.name == "button_press_event":
            self.rect = WindowSelectionRectangle(event, self.plot_axis_z,
                self._onWindowSelected)

    def _onWindowSelected(self, window_start, window_width):
        """
        Function called upon window selection.
        """
        return
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


def launch(event, seismogram_generator, project):
    MisfitGUI(event, seismogram_generator, project)
