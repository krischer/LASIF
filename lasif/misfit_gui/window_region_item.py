#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PyQt4 import QtCore
import pyqtgraph


#TODO FIX this
DEFAULT_AD_SRC_TYPE = "TimeFrequencyPhaseMisfitFichtner2008"

class WindowLinearRegionItem(pyqtgraph.LinearRegionItem):
    def __init__(self, window_group_manager, channel_name, iteration,
                 start, end, event, parent, comm=None, **kwargs):
        """

        :param window: window collection object, this has to change!
        :param event: event_name
        :param parent: plot_widget is the paretn
        :param kwargs: kwargs, not used
        """
        self.win_grp_manager = window_group_manager
        self.channel_name = channel_name
        self.event_time = event["origin_time"]
        self.event_name = event["event_name"]
        self.comm = comm
        self.iteration = iteration

        # Here self.win[0] is given as the real physical time, in my mocking example this therefore does not work
        self.start = start
        self.end = end

        rel_start = self.start - event["origin_time"]
        rel_end = self.end - event["origin_time"]

        values = [rel_start, rel_end]

        super(WindowLinearRegionItem, self).__init__(values=values, **kwargs)
        self._parent = parent
        self._parent.addItem(self)
        self.setZValue(-5)

        self.sigRegionChangeFinished.connect(self.on_region_change_finished)

    def on_region_change_finished(self, *args, **kwargs):
        start, end = args[0].getRegion()
        start = self.event_time + start
        end = self.event_time + end
        self.win_grp_manager.add_window_to_event_channel(self.event_name, self.channel_name,
                                                         start_time=start, end_time=end, weight=1.0)

    def mouseClickEvent(self, ev):
        if ev.modifiers() & (
                QtCore.Qt.AltModifier | QtCore.Qt.ShiftModifier |
                QtCore.Qt.ControlModifier):
            self.plot_adjoint_source()

    def mouseDoubleClickEvent(self, ev):
        if not ev.modifiers():
            self.win_grp_manager.del_window_from_event_channel(
                self.event_name, self.channel_name, self.start, self.end)
            self._parent.removeItem(self)
            ev.accept()

    def plot_adjoint_source(self):
        if self.comm is None:
            raise ValueError("Operation only possible with an active "
                             "communicator instance.")

        import matplotlib.pyplot as plt
        plt.close("all")
        plt.figure(figsize=(15, 10))

        data = self.comm.query.get_matching_waveforms(self.event_name, self.iteration,
                                                      self.channel_name)

        process_params = self.comm.project.preprocessing_params
        self.comm.wins_and_adj_sources.calculate_adjoint_source(
            data=data.data[0], synth=data.synthetics[0], starttime=self.start,
            endtime=self.end, taper="hann",
            taper_percentage=0.05,
            min_period=process_params["highpass_period"],
            max_period=process_params["lowpass_period"],
            ad_src_type="TimeFrequencyPhaseMisfitFichtner2008", plot=True)
        plt.show()

