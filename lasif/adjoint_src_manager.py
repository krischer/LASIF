#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A class handling the adjoint sources.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import numpy as np
import os


class AdjointSourceManager(object):

    """ Class for reading and writing adjoint sources. """

    def __init__(self, directory):
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def _get_tag(self, channel_id, starttime, endtime):
        """
        Helper method returning the filename of the adjoint source.
        """
        return "adjoint_source_%s__%s__%s.npy" % (channel_id, str(starttime),
                                                  str(endtime))

    def write_adjoint_src(self, data, channel_id, starttime, endtime):
        """
        Writes the adjoint sources to a file.
        """
        filename = os.path.join(self.directory, self._get_tag(
            channel_id, starttime, endtime))
        # Save as 64bit floats just to be able to handle any solver and what
        # not.
        np.save(filename, np.require(data, "float64"))

    def get_adjoint_src(self, channel_id, starttime, endtime):
        filename = os.path.join(self.directory, self._get_tag(
            channel_id, starttime, endtime))
        if not os.path.exists(filename):
            return None
        return np.load(filename)
