from PyQt4 import QtCore, QtGui

import pyqtgraph


class WindowLinearRegionItem(pyqtgraph.LinearRegionItem):
    def __init__(self, window, event, **kwargs):
        self.win = window
        start = self.win.starttime - event["origin_time"]
        end = self.win.endtime - event["origin_time"]

        values = [start, end]

        super(WindowLinearRegionItem, self).__init__(values=values, **kwargs)

    def mouseClickEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.accept()
            return
        if not hasattr(self, "menu"):
            self.menu = QtGui.QMenu()
            self.menu.setTitle("Window")

            green = QtGui.QAction("Delete", self.menu)
            #green.triggered.connect(self.setGreen)
            self.menu.addAction(green)
            self.menu.green = green

            alpha = QtGui.QWidgetAction(self.menu)
            alphaSlider = QtGui.QSlider()
            alphaSlider.setOrientation(QtCore.Qt.Horizontal)
            alphaSlider.setMaximum(255)
            alphaSlider.setValue(255)
            #alphaSlider.valueChanged.connect(self.setAlpha)
            alpha.setDefaultWidget(alphaSlider)
            self.menu.addAction(alpha)
            self.menu.alpha = alpha
            self.menu.alphaSlider = alphaSlider

        pos = ev.screenPos()
        self.menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        ev.accept()
