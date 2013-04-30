#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DESCRIPTION

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import inspect
import numpy as np
import os
from scipy.io import loadmat
import unittest

from lasif.adjoint_sources import utils


class AdjointSourceUtilsTestCase(unittest.TestCase):
    """
    Tests for the utility functionalities.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_matlab_range(self):
        """
        Tests the Matlab range command.
        """
        np.testing.assert_array_equal(utils.matlab_range(0, 5, 1),
            np.arange(6))
        np.testing.assert_array_equal(utils.matlab_range(0, 5.5, 1),
            np.arange(6))
        np.testing.assert_array_equal(utils.matlab_range(0, 4.9, 1),
            np.arange(5))

    def test_dispersive_wavetrain(self):
        """
        Tests the dispersive wavetrain calculation by comparing it to a
        reference solution implemented in Matlab.
        """
        # Load the matlab file.
        matlab_file = os.path.join(self.data_dir,
            "matlab_dispersive_wavetrain_reference_solution.mat")
        matlab_file = loadmat(matlab_file)
        u_matlab = matlab_file["u"][0]
        u0_matlab = matlab_file["u0"][0]
        t, u = utils.get_dispersed_wavetrain()
        np.testing.assert_allclose(u, u_matlab)
        np.testing.assert_allclose(t, np.arange(901))
        t0, u0 = utils.get_dispersed_wavetrain(a=3.91, b=0.87, c=0.8,
            body_wave_factor=0.015, body_wave_freq_scale=1.0 / 2.2)
        np.testing.assert_allclose(u0, u0_matlab)
        np.testing.assert_allclose(t0, np.arange(901))

    def test_cross_correlation(self):
        """
        Tests the cross correlation function and compares it to a reference
        solution calculated in Matlab.
        """
        # Load the matlab file.
        matlab_file = os.path.join(self.data_dir,
            "matlab_cross_correlation_reference_solution.mat")
        cc_matlab = loadmat(matlab_file)["cc"][0]

        # Calculate two test signals.
        _, u = utils.get_dispersed_wavetrain()
        _, u0 = utils.get_dispersed_wavetrain(a=3.91, b=0.87, c=0.8,
            body_wave_factor=0.015, body_wave_freq_scale=1.0 / 2.2)

        cc = utils.cross_correlation(u, u0)
        np.testing.assert_allclose(cc, cc_matlab)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(AdjointSourceUtilsTestCase, "test"))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
