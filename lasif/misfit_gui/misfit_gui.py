#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from PyQt4 import QtGui
from PyQt4.QtCore import pyqtSlot
import pyqtgraph as pg
# Default to antialiased drawing.
pg.setConfigOptions(antialias=True, foreground=(50, 50, 50), background=None)


from glob import iglob
import imp
import inspect
import matplotlib.patheffects as PathEffects
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
import os
import random
import sys

from ..colors import COLORS
from .window_region_item import WindowLinearRegionItem

import lasif.visualization


taupy_model = TauPyModel("ak135")


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
            print "Compiling ui file: %s" % ui_file
            with open(py_ui_file, 'w') as open_file:
                uic.compileUi(ui_file, open_file)
        # Import the (compiled) file.
        try:
            import_name = os.path.splitext(os.path.basename(py_ui_file))[0]
            globals()[import_name] = imp.load_source(import_name, py_ui_file)
        except ImportError, e:
            print "Error importing %s" % py_ui_file
            print e.message


path_effects = [PathEffects.withStroke(linewidth=5, foreground="white")]


class Window(QtGui.QMainWindow):
    def __init__(self, comm):
        QtGui.QMainWindow.__init__(self)
        self.comm = comm
        self.ui = qt_window.Ui_MainWindow()  # NOQA
        self.ui.setupUi(self)

        # Set up the map.
        self.map_figure = self.ui.mapView.fig
        self.map_ax = self.map_figure.add_axes([0.0, 0.0, 1.0, 1.0])
        self.basemap = self.comm.project.domain.plot(ax=self.map_ax)
        self._draw()

        # State of the map objects.
        self.current_mt_patches = []

        self.current_window_manager = None

        self.ui.status_label = QtGui.QLabel("")
        self.ui.statusbar.addPermanentWidget(self.ui.status_label)

        self.ui.iteration_selection_comboBox.addItems(
            self.comm.iterations.list())
        for component in ["z", "n", "e"]:
            p = getattr(self.ui, "%s_graph" % component)
            # p.setBackground(None)
            # for d in ("left", "bottom", "top", "right"):
            #     p.getAxis(d).setPen("#333")
            p.setLabel("left", "Velocity", units="m/s")
            p.setLabel("bottom", "Time since event", units="s")

            label = {"z": "vertical", "e": "east", "n": "north"}
            p.setTitle(label[component].capitalize() + " component")

        # Hack to get a proper legend.
        self.ui.e_graph.addLegend(offset=(-2, 2))
        self.ui.e_graph.plot([0], [0], pen="k", name="Data")
        self.ui.e_graph.plot([0], [0], pen="r", name="Synthetics")
        self.ui.e_graph.clear()

    def _draw(self):
        self.map_figure.canvas.draw()

    def _reset_all_plots(self):
        for component in ["z", "n", "e"]:
            p = getattr(self.ui, "%s_graph" % component)
            p.clear()
            p.setXRange(-1, 1)
            p.setYRange(-1, 1)

    @property
    def current_iteration(self):
        return str(self.ui.iteration_selection_comboBox.currentText())

    @property
    def current_event(self):
        return str(self.ui.event_selection_comboBox.currentText())

    @property
    def current_station(self):
        cur_item = self.ui.stations_listWidget.currentItem()
        if cur_item is None:
            return None
        return str(cur_item.text())

    @pyqtSlot(str)
    def on_iteration_selection_comboBox_currentIndexChanged(self, value):
        value = str(value).strip()
        if not value:
            return
        it = self.comm.iterations.get(value)
        self.ui.event_selection_comboBox.setEnabled(True)
        self.ui.event_selection_comboBox.clear()
        self.ui.event_selection_comboBox.addItems(sorted(it.events.keys()))

        if it.scale_data_to_synthetics:
            self.ui.status_label.setText("Data scaled to synthetics for "
                                         "this iteration")
        else:
            self.ui.status_label.setText("")

    def _update_raypath(self, coordinates):
        if hasattr(self, "_current_raypath") and self._current_raypath:
            for _i in self._current_raypath:
                _i.remove()

        event_info = self.comm.events.get(self.current_event)
        self._current_raypath = self.basemap.drawgreatcircle(
            event_info["longitude"], event_info["latitude"],
            coordinates["longitude"], coordinates["latitude"],
            color=COLORS[random.randint(0, len(COLORS) - 1)],
            lw=2, alpha=0.8, zorder=10, path_effects=path_effects)
        self._draw()

    def _update_event_map(self):
        for i in self.current_mt_patches:
            i.remove()

        event = self.comm.events.get(self.current_event)

        self.current_mt_patches = lasif.visualization.plot_events(
            events=[event], map_object=self.basemap, beachball_size=0.04)

        try:
            self.current_station_scatter.remove()
        except:
            pass

        stations = self.comm.query.get_all_stations_for_event(
            self.current_event)

        # Plot the stations. This will also plot raypaths.
        self.current_station_scatter = lasif.visualization \
            .plot_stations_for_event(map_object=self.basemap,
                                     color="0.2", alpha=0.4,
                                     station_dict=stations,
                                     event_info=event, raypaths=False)
        self.map_ax.set_title("No matter the projection, North for the "
                              "moment tensors is always up.")

        if hasattr(self, "_current_raypath") and self._current_raypath:
            for _i in self._current_raypath:
                _i.remove()
            self._current_raypath = []

        self._draw()

    @pyqtSlot(str)
    def on_event_selection_comboBox_currentIndexChanged(self, value):
        value = str(value).strip()
        if not value:
            return
        it = self.comm.iterations.get(self.current_iteration)
        self.ui.stations_listWidget.clear()
        self.ui.stations_listWidget.addItems(
            sorted(it.events[value]["stations"].keys()))

        self.current_window_manager = self.comm.windows.get(
            self.current_event, self.current_iteration)

        self._reset_all_plots()
        self._update_event_map()

    def _window_region_callback(self, *args, **kwargs):
        start, end = args[0].getRegion()
        win = args[0].window_object
        event_starttime = args[0].event_starttime
        start = event_starttime + start
        end = event_starttime + end

        win.starttime = start
        win.endtime = end
        win._Window__collection.write()

    def on_stations_listWidget_currentItemChanged(self, current, previous):
        if current is None:
            return

        self._reset_all_plots()

        try:
            wave = self.comm.query.get_matching_waveforms(
                self.current_event, self.current_iteration,
                self.current_station)
        except Exception as e:
            for component in ["Z", "N", "E"]:
                plot_widget = getattr(self.ui, "%s_graph" % component.lower())
                plot_widget.addItem(pg.TextItem(
                    text=str(e), anchor=(0.5, 0.5),
                    color=(200, 0, 0)))
            return

        event = self.comm.events.get(self.current_event)

        great_circle_distance = locations2degrees(
            event["latitude"], event["longitude"],
            wave.coordinates["latitude"], wave.coordinates["longitude"])
        tts = taupy_model.get_travel_times(
            source_depth_in_km=event["depth_in_km"],
            distance_in_degree=great_circle_distance)

        windows_for_station = \
            self.current_window_manager.get_windows_for_station(
                self.current_station)

        for component in ["Z", "N", "E"]:
            plot_widget = getattr(self.ui, "%s_graph" % component.lower())
            data_tr = [tr for tr in wave.data
                       if tr.stats.channel[-1].upper() == component]
            if data_tr:
                tr = data_tr[0]
                plot_widget.data_id = tr.id
                times = tr.times()
                plot_widget.plot(times, tr.data, pen="k")
            else:
                plot_widget.data_id = None
            synth_tr = [_i for _i in wave.synthetics
                        if _i.stats.channel[-1].upper() == component]
            if synth_tr:
                tr = synth_tr[0]
                times = tr.times()
                plot_widget.plot(times, tr.data, pen="r", )

            if data_tr or synth_tr:
                for tt in tts:
                    if tt.time >= times[-1]:
                        continue
                    if tt.name[0].lower() == "p":
                        pen = "#008c2866"
                    else:
                        pen = "#95000066"
                    plot_widget.addLine(x=tt.time, pen=pen, z=-10)

            plot_widget.autoRange()

            window = [_i for _i in windows_for_station
                      if _i.channel_id[-1].upper() == component]
            if window:
                plot_widget.windows = window[0]
                for win in window[0].windows:
                    WindowLinearRegionItem(win, event, parent=plot_widget)

        self._update_raypath(wave.coordinates)

    def on_reset_view_Button_released(self):
        for component in ["Z", "N", "E"]:
            getattr(self.ui, "%s_graph" % component.lower()).autoRange()

    def __add_window_to_plot_widget(self, plot_widget, x_1, x_2):
        id = plot_widget.data_id
        if id is None:
            QtGui.QMessageBox.information(
                self, "", "Can only create windows if data is available.")
            return

        event = self.comm.events.get(self.current_event)

        window = self.current_window_manager.get(id)
        window.add_window(
            starttime=event["origin_time"] + x_1,
            endtime=event["origin_time"] + x_2,
            weight=1.0
        )
        window.write()

        self.on_stations_listWidget_currentItemChanged(True, False)

    def _add_window(self, origin, min_x, max_x):
        self.__add_window_to_plot_widget(origin, min_x, max_x)

    def on_next_Button_released(self):
        st = self.ui.stations_listWidget
        idx = st.currentIndex().row() + 1
        if idx >= st.count():
            return
        st.setCurrentRow(idx)

    def on_previous_Button_released(self):
        st = self.ui.stations_listWidget
        idx = st.currentIndex().row() - 1
        if idx < 0:
            return
        st.setCurrentRow(idx)

    def on_delete_all_Button_released(self):
        for component in ["Z", "N", "E"]:
            plot_widget = getattr(self.ui, "%s_graph" % component.lower())
            if not hasattr(plot_widget, "windows"):
                continue
            plot_widget.windows.windows[:] = []
            plot_widget.windows.write()
        self.on_stations_listWidget_currentItemChanged(True, False)

    def on_autoselect_Button_released(self):
        windows_for_station = \
            self.current_window_manager.get_windows_for_station(
                self.current_station)
        if windows_for_station:
            QtGui.QMessageBox.information(
                self, "", "Autoselection only works if no windows exists for "
                          "the station.")
            return

        self.comm.actions.select_windows_for_station(self.current_event,
                                                     self.current_iteration,
                                                     self.current_station)
        self.on_stations_listWidget_currentItemChanged(True, False)


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
