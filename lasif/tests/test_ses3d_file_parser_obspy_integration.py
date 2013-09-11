#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the integration of the file parser into ObsPy.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GXU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import numpy as np
import obspy
import os
import unittest


class SES3DFileParserObsPyIntegrationTestCase(unittest.TestCase):
    """
    Test case that checks if the integration into ObsPy as a plug-in works as
    expected.

    This can only work if the module is properly installed.

    Currently contains only one test as the rest is tested in the main ses3d
    file parser test suite and would be redundant.
    """
    def setUp(self):
        # Most generic way to get the actual data directory.
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_ReadingByCheckingReturnedData(self):
        """
        Reads the files with obspy and checks the data.
        """
        filename_theta = os.path.join(self.data_dir, "File_theta")
        filename_phi = os.path.join(self.data_dir, "File_phi")
        filename_r = os.path.join(self.data_dir, "File_r")

        tr_theta = obspy.read(filename_theta)[0]
        self.assertEqual(tr_theta.stats.channel, "X")
        self.assertTrue(hasattr(tr_theta.stats, "ses3d"))
        theta_data = np.array([
            4.23160685E-07, 3.80973177E-07, 3.39335969E-07, 2.98305707E-07,
            2.57921158E-07, 2.18206054E-07, 1.79171423E-07, 1.40820376E-07,
            1.03153077E-07, 6.61708626E-08])
        np.testing.assert_almost_equal(tr_theta.data[-10:], theta_data)

        tr_phi = obspy.read(filename_phi)[0]
        self.assertEqual(tr_phi.stats.channel, "Y")
        self.assertTrue(hasattr(tr_theta.stats, "ses3d"))
        phi_data = np.array([
            4.23160685E-07, 3.80973177E-07, 3.39335969E-07, 2.98305707E-07,
            2.57921158E-07, 2.18206054E-07, 1.79171423E-07, 1.40820376E-07,
            1.03153077E-07, 6.61708626E-08])
        np.testing.assert_almost_equal(tr_phi.data[-10:], phi_data)

        tr_r = obspy.read(filename_r)[0]
        self.assertEqual(tr_r.stats.channel, "Z")
        self.assertTrue(hasattr(tr_theta.stats, "ses3d"))
        r_data = np.array([
            3.33445854E-07, 3.32186886E-07, 3.32869206E-07, 3.35317537E-07,
            3.39320707E-07, 3.44629825E-07, 3.50957549E-07, 3.57983453E-07,
            3.65361842E-07, 3.72732785E-07])
        np.testing.assert_almost_equal(tr_r.data[-10:], r_data)


def suite():
    return unittest.makeSuite(SES3DFileParserObsPyIntegrationTestCase, "test")


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
