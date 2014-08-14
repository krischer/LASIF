#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PyQt4 import QtGui
from PyQt4.QtCore import pyqtSlot
import pyqtgraph as pg
# Default to antialiased drawing.
pg.setConfigOptions(antialias=True)


from glob import iglob
import imp
import inspect
import numpy as np
from obspy.core.util.geodetics import locations2degrees
from obspy.taup.taup import getTravelTimes
import os
import sys


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


class Window(QtGui.QMainWindow):
    def __init__(self, comm):
        QtGui.QMainWindow.__init__(self)
        self.comm = comm
        self.ui = qt_window.Ui_MainWindow()
        self.ui.setupUi(self)

        self.current_window_manager = None

        self.ui.status_label = QtGui.QLabel("")
        self.ui.statusbar.addPermanentWidget(self.ui.status_label)

        self.ui.iteration_selection_comboBox.addItems(
            self.comm.iterations.list())

        for component in ["z", "n", "e"]:
            p = getattr(self.ui, "%s_graph" % component)
            p.setBackground(None)
            for d in ("left", "bottom", "top", "right"):
                p.getAxis(d).setPen("#333")
            p.setLabel("left", "Velocity", units="m/s")
            p.setLabel("bottom", "Time since event", units="s")

            label = {"z": "vertical", "e": "east", "n": "north"}
            p.setTitle(label[component].capitalize() + " component")

        # Hack to get a proper legend.
        self.ui.e_graph.addLegend(offset=(-2, 2))
        self.ui.e_graph.plot([0], [0], pen="k", name="Data")
        self.ui.e_graph.plot([0], [0], pen="r", name="Synthetics")
        self.ui.e_graph.clear()

    def _reset_all_plots(self):
        for component in ["z", "n", "e"]:
            p = getattr(self.ui, "%s_graph" % component)
            p.clear()
            p.autoRange()

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
        self.ui.event_selection_comboBox.addItems(it.events.keys())

        if it.scale_data_to_synthetics:
            self.ui.status_label.setText("Data scaled to synthetics for "
                                         "iteration")
        else:
            self.ui.status_label.setText("")

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

    def on_stations_listWidget_currentItemChanged(self, current, previous):
        if current is None:
            return
        wave = self.comm.query.get_matching_waveforms(
            self.current_event, self.current_iteration, self.current_station)

        event = self.comm.events.get(self.current_event)

        great_circle_distance = locations2degrees(
            event["latitude"], event["longitude"],
            wave.coordinates["latitude"], wave.coordinates["longitude"])
        tts = getTravelTimes(great_circle_distance, event["depth_in_km"],
                             model="ak135")

        windows_for_station = \
            self.current_window_manager.get_windows_for_station(
                self.current_station)

        self._reset_all_plots()

        for component in ["Z", "N", "E"]:
            plot_widget = getattr(self.ui, "%s_graph" % component.lower())
            data_tr = [tr for tr in wave.data
                       if tr.stats.channel[-1].upper() == component]
            if data_tr:
                tr = data_tr[0]
                times = tr.times()
                plot_widget.plot(times, tr.data, pen="k")
            synth_tr = [tr for tr in wave.synthetics
                        if tr.stats.channel[-1].upper() == component]
            if synth_tr:
                tr = synth_tr[0]
                times = tr.times()
                plot_widget.plot(times, tr.data, pen="r")

            if data_tr or synth_tr:
                for tt in tts:
                    if tt["time"] >= times[-1]:
                        continue
                    if tt["phase_name"][0].lower() == "p":
                        pen = "#008c2866"
                    else:
                        pen = "#95000066"
                    plot_widget.addLine(x=tt["time"], pen=pen, z=-10)

            window = [_i for _i in windows_for_station
                      if _i.channel_id[-1].upper() == component]
            if window:
                for win in window[0].windows:
                    start = win.starttime - event["origin_time"]
                    end = win.endtime - event["origin_time"]
                    lr = pg.LinearRegionItem([start,end])
                    lr.setZValue(-5)
                    plot_widget.addItem(lr)


            plot_widget.autoRange()


    def on_reset_view_Button_released(self):
        for component in ["Z", "N", "E"]:
            getattr(self.ui, "%s_graph" % component.lower()).autoRange()

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


def launch(comm):
    # Automatically compile all ui files if they have been changed.
    compile_and_import_ui_files()

    # Launch and open the window.
    app = QtGui.QApplication(sys.argv, QtGui.QApplication.GuiClient)
    window = Window(comm)

    # Show and bring window to foreground.
    window.show()
    window.raise_()
    os._exit(app.exec_())
