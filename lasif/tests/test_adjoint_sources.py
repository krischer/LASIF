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
import numpy as np
import unittest

from lasif.adjoint_sources import utils


class AdjointSourceUtilsTestCase(unittest.TestCase):
    """
    Tests for the utility functionalities.
    """
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


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(AdjointSourceUtilsTestCase, "test"))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
