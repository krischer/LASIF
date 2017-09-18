#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the domain definitions in LASIF.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import absolute_import

import copy

from lasif import domain
from .testing_helpers import images_are_identical, reset_matplotlib


def setup_function(function):
    """
    Reset matplotlib.
    """
    reset_matplotlib()


# def test_global_domain_point_in_domain():
#     """
#     Trivial test...
#     """
#     d = domain.GlobalDomain()
#     assert d.point_in_domain(0, 0)
#     assert d.point_in_domain(-90, +90)
#     assert d.point_in_domain(0, 180)

#
# def test_plotting_global_domain(tmpdir):
#     """
#     Tests the plotting of a global domain.
#     """
#     #assert False, tmpdir
#     #domain.GlobalDomain().plot(plot_simulation_domain=True)
#     #images_are_identical("domain_global", str(tmpdir))
