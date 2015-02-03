#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple class dealing with different 1D Models.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import inspect
import numpy as np
import os

data_dir = \
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

MODELS = {"ak135-f": os.path.join(data_dir, "ak135f.txt")}

EXPOSED_VALUES = ["depth_in_km", "density", "vp", "vs", "Q_kappa", "Q_mu"]


class OneDimensionalModel(object):
    """
    Simple class dealing with 1D earth models.
    """
    def __init__(self, model_name):
        """
        :param model_name: The name of the used model. Possible names are:
            'ak135-F'
        """
        if model_name.lower() == "ak135-f":
            self._read_ak135f()
        else:
            msg = "Unknown model '%s'. Possible models: %s" % (
                model_name, ", ".join(MODELS.keys()))
            raise ValueError(msg)

    def _read_ak135f(self):
        data = np.loadtxt(MODELS["ak135-f"], comments="#")

        self._depth_in_km = data[:, 0]
        self._density = data[:, 1]
        self._vp = data[:, 2]
        self._vs = data[:, 3]
        self._Q_kappa = data[:, 4]
        self._Q_mu = data[:, 5]

    def get_value(self, value_name, depth):
        """
        Returns a value at a requested depth. Currently does a simple linear
        interpolation between the two closest values.
        """
        if value_name not in EXPOSED_VALUES:
            msg = "'%s' is not a valid value name. Valid names: %s" % \
                (value_name, ", ".join(EXPOSED_VALUES))
            raise ValueError(msg)
        return np.interp(depth, self._depth_in_km,
                         getattr(self, "_" + value_name))
