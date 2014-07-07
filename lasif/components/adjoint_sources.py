#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .component import Component


class AdjointSourcesComponent(Component):
    """
    Component dealing with the adjoint sources.

    :param ad_src_folder: The folder where the adjoint sources are stored.
    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, ad_src_folder, communicator, component_name):
        self._folder = ad_src_folder
        super(AdjointSourcesComponent, self).__init__(
            communicator, component_name)
