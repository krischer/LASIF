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

from lasif import ses3d_models

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
            "model": None
        }
        self.model = None

        # Setup the figures.
        self._setup_figures()

        # Status bar label. Will be used to display the memory consumption.
        self.ui.status_label = QtGui.QLabel("")
        self.ui.statusbar.addPermanentWidget(self.ui.status_label)

        # Add list of all models to the UI.
        self.ui.model_selection_comboBox.addItems(
            [""] + self.comm.models.list())

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
            [0.02, 0.05, 0.40, 0.94], axisbg="none")

        # Histogram.
        self.figures["histogram"] = self.ui.histogram.fig
        self.axes["histogram"] =self.figures["histogram"].add_axes(
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
            self.plot_depths(ret_val["depths"], ret_val["values"])

            # Plot the marker at the correct position.
            self.basemap.scatter(
                [ret_val["longitude"]], [ret_val["latitude"]],
                marker="x", s=100, latlon=True, color="0.2", zorder=200,
                path_effects=[PathEffects.withStroke(linewidth=4,
                                                     foreground="white")])
            self.figures["map"].canvas.draw()

    def plot_depths(self, depths, values):
        ax = self.axes["depth_profile"]
        ax.clear()
        ax.plot(values, depths, color="#55A868", lw=3)
        ax.set_xlabel(self._gui_component)
        ax.set_ylabel("Depth [km]")
        ax.grid(True)
        # if self.current_actual_depth is not None:
        #     a, b = self.depth_profile_ax.get_xlim()
        #     ax.hlines(self.current_actual_depth, a, b, color="red")
        self.figures["depth_profile"].canvas.draw()

    def _is_state_still_valid(self):
        """
        Checks if the current state is still valid.
        """
        state = {
            "component": self._gui_component,
            "depth": self._gui_depth,
            "model": self._gui_model
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

        # Now figure out what changed.
        if self.current_state["model"] != model:
            self._load_new_model()

        self.current_state["component"] = component
        self.current_state["depth"] = depth

        if None in self.current_state.values():
            return

        self.model.parse_component(component)

        # Plot model and colorbar.
        actual_depth, im, depth_data = self.model.plot_depth_slice(
            component, depth, self.basemap)

        if actual_depth is None:
            self.ui.depth_label.setText("Desired Depth: %.1f km" % depth)
            return

        self.ui.depth_label.setText("Desired Depth: %.1f km" % depth)
        self.ui.plotted_depth_label.setText("Plotted Depth: %.1f km" %
                                            actual_depth)

        self.axes["colorbar"].clear()
        cm = self.figures["colorbar"].colorbar(im, cax=self.axes["colorbar"])
        if component in UNIT_DICT:
            cm.set_label(UNIT_DICT[component], fontsize="x-large", rotation=0)

        # Plot the histogram.
        ax = self.axes["histogram"]
        ax.clear()
        ax.hist(depth_data.ravel(), bins=150, color="#C44E52")
        min, max = depth_data.min(), depth_data.max()
        v_range = max - min
        ax.set_xlim(min - v_range * 0.1, max + v_range * 0.1)
        ax.set_yticks([])

        self._draw()

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

        self.model = ses3d_models.RawSES3DModelHandler(
            self.comm.models.get(model), domain=self.comm.project.domain)
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
        self.ui.depth_slider.setRange(*sorted(depths))

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

    @pyqtSlot(int)
    def on_depth_slider_valueChanged(self, value):
        self._update()

    @pyqtSlot(str)
    def on_variable_selection_comboBox_currentIndexChanged(self, value):
        self._update()

    @pyqtSlot(str)
    def on_model_selection_comboBox_currentIndexChanged(self, value):
        self._update()


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
