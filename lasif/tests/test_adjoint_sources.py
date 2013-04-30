#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test cases for everything related to the adjoint source calculations.

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

from lasif.adjoint_sources import utils, time_frequency, ad_src_tf_phase_misfit


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


class TimeFrequencyTestCase(unittest.TestCase):
    """
    Test case for functionality related to time frequency representations.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_time_frequency_transform(self):
        """
        Tests the basic time frequency transformation.
        """
        t, u = utils.get_dispersed_wavetrain()
        tau, nu, tfs = \
            time_frequency.time_frequency_transform(t, u, 2, 10, 0.0)

        # Load the matlab output.
        matlab = os.path.join(self.data_dir,
            "matlab_tfa_output_reference_solution.mat")
        matlab = loadmat(matlab)
        #tau_matlab = matlab["TAU"]
        #nu_matlab = matlab["NU"]
        tfs_matlab = matlab["tfs"]

        # Some tolerance is needed to due numeric differences.
        tolerance = 1E-5
        min_value = np.abs(tfs).max() * tolerance
        tfs[np.abs(tfs) < min_value] = 0 + 0j
        tfs_matlab[np.abs(tfs_matlab) < min_value] = 0 + 0j

        np.testing.assert_allclose(np.abs(tfs), np.abs(tfs_matlab))
        np.testing.assert_allclose(np.angle(tfs), np.angle(tfs_matlab))


class AdjointSourceTestCase(unittest.TestCase):
    """
    Tests the actual adjoint source calculations.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_adjoint_time_frequency_phase_misfit_source(self):
        """
        Tests the adjoint source calculation for the time frequency phase
        misfit after Fichtner et. al. (2008).
        """
        # Load the matlab output.
        ad_src_matlab = os.path.join(self.data_dir,
            "matlab_tf_phase_misfit_adjoint_source_reference_solution.mat")
        ad_src_matlab = loadmat(ad_src_matlab)["ad_src"].transpose()[0]

        # Generate some data.
        t, u = utils.get_dispersed_wavetrain()
        _, u0 = utils.get_dispersed_wavetrain(a=3.91, b=0.87, c=0.8,
            body_wave_factor=0.015, body_wave_freq_scale=1.0 / 2.2)

        adjoint_src = ad_src_tf_phase_misfit.adsrc_tf(t, u, u0, 2, 10, 0.0)
        ad_src = adjoint_src["adjoint_source"]
        # Assert the misfit.
        self.assertAlmostEqual(adjoint_src["misfit"], 0.271417, 5)

        # Some testing tolerance is needed mainly due to the phase being hard
        # to define for small amplitudes.
        tolerance = np.abs(ad_src).max() * 1.2E-3
        np.testing.assert_allclose(ad_src, ad_src_matlab, 1E-7, tolerance)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(AdjointSourceUtilsTestCase, "test"))
    suite.addTest(unittest.makeSuite(TimeFrequencyTestCase, "test"))
    suite.addTest(unittest.makeSuite(AdjointSourceTestCase, "test"))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
