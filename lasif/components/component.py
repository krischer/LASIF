#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .communicator import Communicator


class Component(object):
    """
    Base class of the different components.
    """
    def __init__(self, communicator, component_name):
        if not isinstance(communicator, Communicator):
            raise TypeError
        self.comm = communicator
        self.comm.register(component_name, self)
