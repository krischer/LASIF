#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .component import Component


class WaveformsComponent(Component):
    def __init__(self, data_folder, synthetics_folder, communicator,
                 component_name):
        self._data_folder = data_folder
        self.synthetics_folder = synthetics_folder

        super(WaveformsComponent, self).__init__(communicator, component_name)

    def get_raw(self, event_name):
        pass
