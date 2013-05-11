#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2012
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from matplotlib.patches import Rectangle


class WindowSelectionRectangle(object):
    def __init__(self, event, axis, on_window_selection_callback):
        self.axis = axis
        if event.inaxes != self.axis:
            return
        # Store the axes it has been initialized in.
        self.axes = event.inaxes
        ymin, ymax = self.axes.get_ylim()
        self.min_x = event.xdata
        self.intial_selection_active = True
        self.rect = Rectangle((event.xdata, ymin), 0, ymax - ymin, color="0.3",
            alpha=0.5, edgecolor="0.5")
        self.axes.add_patch(self.rect)
        # Get the canvas.
        self.canvas = self.rect.figure.canvas

        # Use blittig for fast animations.
        self.rect.set_animated(True)
        self.background = self.canvas.copy_from_bbox(self.rect.axes.bbox)

        self._connect()

        self.on_window_selection_callback = on_window_selection_callback

    def __del__(self):
        """
        Disconnect the events upon deallocating.
        """
        self.canvas.mpl_disconnect(self.conn_button_press)
        self.canvas.mpl_disconnect(self.conn_button_release)
        self.canvas.mpl_disconnect(self.conn_mouse_motion)

    def _connect(self):
        """
        Connect to the necessary events.
        """
        self.conn_button_press = self.rect.figure.canvas.mpl_connect(
            'button_press_event', self.on_button_press)
        self.conn_button_release = self.rect.figure.canvas.mpl_connect(
            'button_release_event', self.on_button_release)
        self.conn_mouse_motion = self.rect.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_mouse_motion)

    def on_button_press(self, event):
        pass

    def on_button_release(self, event):
        if event.inaxes != self.axis:
            return

        if event.button != 1:
            return
        # turn off the rect animation property and reset the background
        self.rect.set_animated(False)
        self.background = None

        self.intial_selection_active = False
        self.canvas.draw()

        self.on_window_selection_callback(self.rect.get_x(),
            self.rect.get_width(), self.axis)

    def on_mouse_motion(self, event):
        if event.button != 1 or \
            self.intial_selection_active is not True:
            return
        if event.xdata is not None:
            self.rect.set_width(event.xdata - self.min_x)

        # restore the background region
        self.canvas.restore_region(self.background)
        # redraw just the current rectangle
        self.axes.draw_artist(self.rect)
        # blit just the redrawn area
        self.canvas.blit(self.axes.bbox)
