#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
import joblib
import numpy as np
import os

from lasif import LASIFNotFoundError, LASIFAdjointSourceCalculationError
from .component import Component
from ..adjoint_sources.ad_src_tf_phase_misfit import adsrc_tf_phase_misfit
from ..adjoint_sources.ad_src_l2_norm_misfit import adsrc_l2_norm_misfit
from ..adjoint_sources.ad_src_cc_time_shift import adsrc_cc_time_shift




class AdjointSourcesComponent(Component):
    """
    Component dealing with the adjoint sources.

    :param communicator: The communicator instance.
    :param component_name: The name of this component for the communicator.
    """
    def __init__(self, communicator, component_name):
        super(AdjointSourcesComponent, self).__init__(
            communicator, component_name)

