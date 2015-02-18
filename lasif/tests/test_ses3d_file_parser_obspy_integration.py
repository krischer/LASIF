#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the integration of the file parser into ObsPy.

Test case that checks if the integration into ObsPy as a plug-in works as
expected.

This can only work if the module is properly installed.

Currently contains only one test as the rest is tested in the main ses3d file
parser test suite and would be redundant.

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


data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


def test_ReadingByCheckingReturnedData():
    """
    Reads the files with obspy and checks the data.
    """
    filename_theta = os.path.join(data_dir, "File_theta")
    filename_phi = os.path.join(data_dir, "File_phi")
    filename_r = os.path.join(data_dir, "File_r")

    tr_theta = obspy.read(filename_theta)[0]
    assert tr_theta.stats.channel == "X"
    assert hasattr(tr_theta.stats, "ses3d")
    theta_data = np.array([
        4.23160685E-07, 3.80973177E-07, 3.39335969E-07, 2.98305707E-07,
        2.57921158E-07, 2.18206054E-07, 1.79171423E-07, 1.40820376E-07,
        1.03153077E-07, 6.61708626E-08])
    np.testing.assert_almost_equal(tr_theta.data[-10:], theta_data)

    tr_phi = obspy.read(filename_phi)[0]
    assert tr_phi.stats.channel == "Y"
    assert hasattr(tr_theta.stats, "ses3d")
    phi_data = np.array([
        4.23160685E-07, 3.80973177E-07, 3.39335969E-07, 2.98305707E-07,
        2.57921158E-07, 2.18206054E-07, 1.79171423E-07, 1.40820376E-07,
        1.03153077E-07, 6.61708626E-08])
    np.testing.assert_almost_equal(tr_phi.data[-10:], phi_data)

    tr_r = obspy.read(filename_r)[0]
    assert tr_r.stats.channel == "Z"
    assert hasattr(tr_theta.stats, "ses3d")
    r_data = np.array([
        3.33445854E-07, 3.32186886E-07, 3.32869206E-07, 3.35317537E-07,
        3.39320707E-07, 3.44629825E-07, 3.50957549E-07, 3.57983453E-07,
        3.65361842E-07, 3.72732785E-07])
    np.testing.assert_almost_equal(tr_r.data[-10:], r_data)


def test_reading_headonly():
    """
    Reads the files in headonly mode.
    """
    filename_theta = os.path.join(data_dir, "File_theta")
    filename_phi = os.path.join(data_dir, "File_phi")
    filename_r = os.path.join(data_dir, "File_r")

    tr_theta = obspy.read(filename_theta, headonly=True)[0]
    assert tr_theta.stats.channel == "X"
    assert hasattr(tr_theta.stats, "ses3d")
    assert tr_theta.stats.npts == 3300

    tr_phi = obspy.read(filename_phi, headonly=True)[0]
    assert tr_phi.stats.channel == "Y"
    assert hasattr(tr_phi.stats, "ses3d")
    assert tr_phi.stats.npts == 3300

    tr_r = obspy.read(filename_r, headonly=True)[0]
    assert tr_r.stats.channel == "Z"
    assert hasattr(tr_r.stats, "ses3d")
    assert tr_r.stats.npts == 3300


def test_reading_headonly_compare_to_normal_reading():
    """
    Reads the files in headonly mode by comparing to data read in normal mode.
    """
    filename_theta = os.path.join(data_dir, "File_theta")
    filename_phi = os.path.join(data_dir, "File_phi")
    filename_r = os.path.join(data_dir, "File_r")

    tr_theta = obspy.read(filename_theta)[0]
    tr_theta_h = obspy.read(filename_theta, headonly=True)[0]
    assert tr_theta.stats == tr_theta_h.stats
    assert len(tr_theta.data) == 3300
    assert len(tr_theta_h.data) == 0

    tr_phi = obspy.read(filename_phi)[0]
    tr_phi_h = obspy.read(filename_phi, headonly=True)[0]
    assert tr_phi.stats == tr_phi_h.stats
    assert len(tr_phi.data) == 3300
    assert len(tr_phi_h.data) == 0

    tr_r = obspy.read(filename_r)[0]
    tr_r_h = obspy.read(filename_r, headonly=True)[0]
    assert tr_r.stats == tr_r_h.stats
    assert len(tr_phi.data) == 3300
    assert len(tr_phi_h.data) == 0
