#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PyQt4 import QtCore
import pyqtgraph


class WindowLinearRegionItem(pyqtgraph.LinearRegionItem):
    def __init__(self, window, event, parent, **kwargs):
        self.win = window
        self.event_time = event["origin_time"]
        start = self.win.starttime - event["origin_time"]
        end = self.win.endtime - event["origin_time"]

        values = [start, end]

        super(WindowLinearRegionItem, self).__init__(values=values, **kwargs)
        self._parent = parent
        self._parent.addItem(self)
        self.setZValue(-5)

        self.sigRegionChangeFinished.connect(self.on_region_change_finished)

    def on_region_change_finished(self, *args, **kwargs):
        start, end = args[0].getRegion()
        start = self.event_time + start
        end = self.event_time + end

        self.win.starttime = start
        self.win.endtime = end
        self.win._Window__collection.write()

    def mouseClickEvent(self, ev):
        if ev.modifiers() & (
                QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier |
                QtCore.Qt.ControlModifier):
            self.win.plot_adjoint_source()

    def mouseDoubleClickEvent(self, ev):
        if not ev.modifiers():
            coll = self.win._Window__collection
            coll.delete_window(self.win.starttime, self.win.endtime)
            coll.write()
            self._parent.removeItem(self)
            ev.accept()
