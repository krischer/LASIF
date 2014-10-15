#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import collections
import pyqtgraph
import time

from .window_region_item import WindowLinearRegionItem

Click = collections.namedtuple("Click", ["ev", "time"])

MAX_TIME_BETWEEN_CLICKS = 2.0


class PlotWidget(pyqtgraph.PlotWidget):
    def __init__(self, *args, **kwargs):
        super(PlotWidget, self).__init__(*args, **kwargs)
        self.click_active = False
        self.last_click = None
        self.scene().sigMouseClicked.connect(self.sigMouseReleased)

    def sigMouseReleased(self, ev, *args):
        # Avoid some issues with the other items.
        if isinstance(ev.currentItem, WindowLinearRegionItem):
            self.last_click = None
            return

        t = time.time()
        if self.click_active:
            self.click_active = False
            if not self.last_click or (t - self.last_click.time) > \
                    MAX_TIME_BETWEEN_CLICKS:
                self.click_active = True
                self.last_click = Click(ev, t)
                return
            self.add_box(self.last_click, Click(ev, t))
            self.last_click = None
        else:
            self.click_active = True
            self.last_click = Click(ev, t)

    def add_box(self, c_1, c_2):
        # A certain threshold to guard against double clicks.
        if abs(c_2.ev.pos().x() - c_1.ev.pos().x()) < 25:
            return
        # Send to grandfather which is the Misfit GUI main window.
        x_1 = self.plotItem.vb.mapSceneToView(c_1.ev.scenePos()).x()
        x_2 = self.plotItem.vb.mapSceneToView(c_2.ev.scenePos()).x()
        self.parent().parent()._add_window(self, *sorted([x_1, x_2]))
