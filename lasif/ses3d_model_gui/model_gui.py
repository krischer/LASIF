#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple GUI for plotting binary SES3D models on the GLL grid.

This is NOT an example of proper GUI design but rather a hacked together
special purpose tool that does what I need it to do.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import, print_function

from PyQt4 import QtGui
from PyQt4.QtCore import pyqtSlot

from glob import iglob
import imp
import inspect
import os
import sys
import matplotlib.patheffects as PathEffects

from lasif.colors import COLORS


# Pretty units for some components.
UNIT_DICT = {
    "vp": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "vsv": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "vsh": r"$\frac{\mathrm{km}}{\mathrm{s}}$",
    "rho": r"$\frac{\mathrm{kg}}{\mathrm{m}^3}$",
    "rhoinv": r"$\frac{\mathrm{m}^3}{\mathrm{kg}}$",
}


def compile_and_import_ui_files():
    """
    Automatically compiles all .ui files found in the same directory as the
    application py file.
    They will have the same name as the .ui files just with a .py extension.

    Needs to be defined in the same file as function loading the gui as it
    modifies the globals to be able to automatically import the created py-ui
    files. Its just very convenient.
    """
    directory = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    for filename in iglob(os.path.join(directory, '*.ui')):
        ui_file = filename
        py_ui_file = os.path.splitext(ui_file)[0] + os.path.extsep + 'py'
        if not os.path.exists(py_ui_file) or \
                (os.path.getmtime(ui_file) >= os.path.getmtime(py_ui_file)):
            from PyQt4 import uic
            print("Compiling ui file: %s" % ui_file)
            with open(py_ui_file, 'w') as open_file:
                uic.compileUi(ui_file, open_file)
        # Import the (compiled) file.
        try:
            import_name = os.path.splitext(os.path.basename(py_ui_file))[0]
            globals()[import_name] = imp.load_source(import_name, py_ui_file)
        except ImportError, e:
            print("Error importing %s" % py_ui_file)
            print(e.message)


class Window(QtGui.QMainWindow):
    def __init__(self, comm):
        QtGui.QMainWindow.__init__(self)
        # LASIF communicator instances.
        self.comm = comm

        # Setup the GUI.
        self.ui = ses3d_model_gui.Ui_MainWindow()  # NOQA
        self.ui.setupUi(self)

        # Lazy way to avoid circular signals. This is just a very simply GUI
        # so this should be good enough.
        self.current_state = {
            "component": None,
            "depth": None,
            "model": None,
            "style": None,
        }
        self.model = None

        # Keeping track of random plotted stuff.
        self.plot_state = {
            "depth_markers": [],
            "depth_marker_count": 0,
            "actual_depth": 0,
            "depth_profile_line": []
        }

        # Setup the figures.
        self._setup_figures()

        # Status bar label. Will be used to display the memory consumption.
        self.ui.status_label = QtGui.QLabel("")
        self.ui.statusbar.addPermanentWidget(self.ui.status_label)

        # Add list of all models and kernels to the UI.
        # XXX: Do this with a radio box or tab or something else...this is
        # really ugly right now.
        all_models = ["MODEL: %s" % _i for _i in self.comm.models.list()]
        all_kernels = [
            "KERNEL: %s | %s" % (_i["iteration"], _i["event"])
            for _i in self.comm.kernels.list()]

        self.ui.model_selection_comboBox.addItems(
            [""] + all_models + all_kernels)

    def _setup_figures(self):
        """
        Setup all the figures.
        """
        self.figures = {}
        self.axes = {}

        # Map
        self.figures["map"] = self.ui.mapView.fig
        self.axes["map"] = self.figures["map"].add_axes(
            [0.01, 0.01, 0.98, 0.98], axisbg="none")

        # Colorbar
        self.figures["colorbar"] = self.ui.colorbar.fig
        self.axes["colorbar"] = self.figures["colorbar"].add_axes(
            [0.02, 0.05, 0.40, 0.90], axisbg="none")

        # Histogram.
        self.figures["histogram"] = self.ui.histogram.fig
        self.axes["histogram"] = self.figures["histogram"].add_axes(
            [0.05, 0.1, 0.9, 0.85], axisbg="none")
        # Only show the bottom.
        self.axes["histogram"].spines["right"].set_visible(False)
        self.axes["histogram"].spines["left"].set_visible(False)
        self.axes["histogram"].spines["top"].set_visible(False)
        self.axes["histogram"].xaxis.set_ticks_position("bottom")
        self.axes["histogram"].set_yticks([])

        # Depth Profile.
        self.figures["depth_profile"] = self.ui.depth_profile.fig
        self.axes["depth_profile"] = self.figures["depth_profile"].add_axes(
            [0.25, 0.08, 0.7, 0.89], axisbg='none')
        self.axes["depth_profile"].invert_yaxis()

        for axis in self.axes.values():
            axis.clear()

        for figure in self.figures.values():
            figure.set_facecolor('none')

        # Plot the map.
        self.basemap = self.comm.project.domain.plot(ax=self.axes["map"])

        # Connect event to be able the generate depth profiles.
        self.figures["map"].canvas.mpl_connect('button_press_event',
                                               self._on_map_fig_button_press)

        # Redraw all figures.
        self._draw()

    def _draw(self):
        """
        Draw the canvases.
        """
        for fig in self.figures.values():
            fig.canvas.draw()

    def _clear_current_state(self):
        """
        Clears the current state of the app.
        """
        for key in self.current_state:
            self.current_state[key] = None

    def _on_map_fig_button_press(self, event):
        """
        Fired when the map is clicked.
        """
        if event.button != 1 or not event.inaxes or not self.model:
            return

        lon, lat = self.basemap(event.xdata, event.ydata, inverse=True)

        if self.model:
            ret_val = self.model.get_depth_profile(
                self._gui_component, longitude=lon, latitude=lat)
            color = COLORS[self.plot_state["depth_marker_count"] % len(COLORS)]
            self.plot_depths(ret_val["depths"], ret_val["values"], color=color)

            # Plot the marker at the correct position.
            self.plot_state["depth_markers"].extend(self.basemap.plot(
                [ret_val["longitude"]], [ret_val["latitude"]],
                marker="x", ms=15, mew=3, latlon=True,
                color=color,
                zorder=200, path_effects=[PathEffects.withStroke(
                    linewidth=5, foreground="white")]))
            self.figures["map"].canvas.draw()
            self.plot_state["depth_marker_count"] += 1

    def plot_depths(self, depths, values, color):
        ax = self.axes["depth_profile"]
        ax.plot(values, depths, color=color, lw=3, alpha=0.8, zorder=10)
        ax.set_xlabel(self._gui_component)
        ax.set_ylabel("Depth [km]")
        ax.grid(True)
        ax.set_ylim(sorted(self.model.depth_bounds, reverse=True))
        self.figures["depth_profile"].canvas.draw()

    def _is_state_still_valid(self):
        """
        Checks if the current state is still valid.
        """
        state = {
            "component": self._gui_component,
            "depth": self._gui_depth,
            "model": self._gui_model,
            "style": self._gui_style
        }
        return state == self.current_state

    def _update(self):
        """
        Fired any time a UI event is triggered that might change the plotted
        model or component.
        """
        if self._is_state_still_valid():
            return

        model = self._gui_model
        depth = self._gui_depth
        component = self._gui_component
        style = self._gui_style

        # Now figure out what changed.
        if self.current_state["model"] != model:
            self._load_new_model()

        self.current_state["component"] = component
        self.current_state["depth"] = depth
        self.current_state["style"] = style

        if None in self.current_state.values():
            return

        self.model.parse_component(component)

        # Plot model and colorbar.
        ret_val = self.model.plot_depth_slice(
            component, depth, self.basemap,
            absolute_values=True if style == "absolute" else False)

        if ret_val is None:
            self.ui.depth_label.setText("Desired Depth: %.1f km" % depth)
            return

        self.ui.depth_label.setText("Desired Depth: %.1f km" % depth)
        self.ui.plotted_depth_label.setText("Plotted Depth: %.1f km" %
                                            ret_val["depth"])
        self.plot_state["actual_depth"] = ret_val["depth"]
        self.set_depth_profile_line()

        self.axes["colorbar"].clear()
        cm = self.figures["colorbar"].colorbar(
            ret_val["mesh"], cax=self.axes["colorbar"])
        if style == "relative" and component in ["rho", "vsv", "vsh", "vp"]:
            cm.set_label("% diff to AK135", fontsize="small", rotation=270)
        elif component in UNIT_DICT:
            cm.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)

        # Plot a histogram of the value distribution for the current depth
        # slice.
        ax = self.axes["histogram"]
        ax.clear()
        ax.hist(ret_val["data"].ravel(), bins=150, color=COLORS[2])
        _min, _max = ret_val["data"].min(), ret_val["data"].max()
        v_range = max(abs(_max - _min), 0.1 * abs(_max))
        ax.set_xlim(_min - v_range * 0.1, _max + v_range * 0.1)
        ax.set_yticks([])

        self._draw()

    def set_depth_profile_line(self):
        for _i in self.plot_state["depth_profile_line"]:
            try:
                _i.remove()
            except:
                pass
            del _i
        self.plot_state["depth_profile_line"] = []

        if self.plot_state["actual_depth"] is not None:
            ax = self.axes["depth_profile"]
            a, b = ax.get_xlim()
            self.plot_state["depth_profile_line"].append(
                ax.hlines(self.plot_state["actual_depth"], a, b, color="0.2"))

        self.figures["depth_profile"].canvas.draw()

    def _load_new_model(self):
        """
        Loads a new model.
        """
        self.ui.variable_selection_comboBox.clear()

        model = self._gui_model
        if not model:
            self.model = None
            self.current_state["model"] = model
            return

        if model.startswith("MODEL:"):
            m = model.strip().lstrip("MODEL:").strip()
            self.model = self.comm.models.get_model_handler(m)
        elif model.startswith("KERNEL:"):
            kernel = model.strip().lstrip("KERNEL:").strip()
            iteration, event = kernel.split("|")
            iteration = iteration.strip()
            event = event.strip()
            self.model = self.comm.kernels.get_model_handler(
                iteration=iteration, event=event)
        else:
            raise NotImplementedError

        self.current_state["model"] = model

        self.ui.variable_selection_comboBox.clear()
        self.ui.variable_selection_comboBox.addItems(sorted(
            self.model.available_derived_components
        ))
        self.ui.variable_selection_comboBox.insertSeparator(len(
            self.model.available_derived_components))
        self.ui.variable_selection_comboBox.addItems(sorted(
            self.model.components.keys()))

        depths = sorted(self.model.depth_bounds)
        self.ui.depth_slider.setRange(*depths)

    @property
    def _gui_component(self):
        comp = str(self.ui.variable_selection_comboBox.currentText()).strip()
        return comp if comp else None

    @property
    def _gui_depth(self):
        return float(self.ui.depth_slider.value())

    @property
    def _gui_model(self):
        mod = str(self.ui.model_selection_comboBox.currentText()).strip()
        return mod if mod else None

    @property
    def _gui_style(self):
        if self.ui.radio_button_absolute.isChecked():
            return "absolute"
        return "relative"

    @pyqtSlot()
    def on_radio_button_absolute_clicked(self):
        self._update()

    @pyqtSlot()
    def on_radio_button_relative_clicked(self):
        self._update()

    @pyqtSlot(int)
    def on_depth_slider_valueChanged(self, value):
        self._update()

    @pyqtSlot(str)
    def on_variable_selection_comboBox_currentIndexChanged(self, value):
        self._update()

    @pyqtSlot(str)
    def on_model_selection_comboBox_currentIndexChanged(self, value):
        self._update()

    @pyqtSlot()
    def on_clear_profiles_button_clicked(self):
        for m in self.plot_state["depth_markers"]:
            m.remove()
            del m
        self.plot_state["depth_markers"] = []
        self.axes["depth_profile"].clear()
        self.set_depth_profile_line()
        self._draw()


def launch(comm):
    # Automatically compile all ui files if they have been changed.
    compile_and_import_ui_files()

    # Launch and open the window.
    app = QtGui.QApplication(sys.argv, QtGui.QApplication.GuiClient)
    window = Window(comm)

    # Move window to center of screen.
    window.move(
        app.desktop().screen().rect().center() - window.rect().center())
    # Show and bring window to foreground.
    window.show()
    window.raise_()
    os._exit(app.exec_())
