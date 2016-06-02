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

import obspy
from scipy.io import loadmat

from lasif.adjoint_sources import utils, time_frequency, ad_src_tf_phase_misfit

from .testing_helpers import images_are_identical, reset_matplotlib


data_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))), "data")


def setup_function(function):
    """
    Reset matplotlib.
    """
    reset_matplotlib()


def test_matlab_range():
    """
    Tests the Matlab range command.
    """
    np.testing.assert_array_equal(utils.matlab_range(0, 5, 1), np.arange(6))
    np.testing.assert_array_equal(utils.matlab_range(0, 5.5, 1), np.arange(6))
    np.testing.assert_array_equal(utils.matlab_range(0, 4.9, 1), np.arange(5))


def test_dispersive_wavetrain():
    """
    Tests the dispersive wavetrain calculation by comparing it to a
    reference solution implemented in Matlab.
    """
    # Load the matlab file.
    matlab_file = os.path.join(
        data_dir, "matlab_dispersive_wavetrain_reference_solution.mat")
    matlab_file = loadmat(matlab_file)
    u_matlab = matlab_file["u"][0]
    u0_matlab = matlab_file["u0"][0]
    t, u = utils.get_dispersed_wavetrain()
    np.testing.assert_allclose(u, u_matlab)
    np.testing.assert_allclose(t, np.arange(901))
    t0, u0 = utils.get_dispersed_wavetrain(
        a=3.91, b=0.87, c=0.8, body_wave_factor=0.015,
        body_wave_freq_scale=1.0 / 2.2)
    np.testing.assert_allclose(u0, u0_matlab)
    np.testing.assert_allclose(t0, np.arange(901))


def test_cross_correlation():
    """
    Tests the cross correlation function and compares it to a reference
    solution calculated in Matlab.
    """
    # Load the matlab file.
    matlab_file = os.path.join(
        data_dir, "matlab_cross_correlation_reference_solution.mat")
    cc_matlab = loadmat(matlab_file)["cc"][0]

    # Calculate two test signals.
    _, u = utils.get_dispersed_wavetrain()
    _, u0 = utils.get_dispersed_wavetrain(
        a=3.91, b=0.87, c=0.8, body_wave_factor=0.015,
        body_wave_freq_scale=1.0 / 2.2)

    cc = utils.cross_correlation(u, u0)
    np.testing.assert_allclose(cc, cc_matlab)


def test_time_frequency_transform():
    """
    Tests the basic time frequency transformation.
    """
    t, u = utils.get_dispersed_wavetrain(dt=2.0)
    tau, nu, tfs = time_frequency.time_frequency_transform(
        t=t, s=u, width=10.0)

    # Load the matlab output.
    matlab = os.path.join(
        data_dir, "matlab_tfa_output_reference_solution.mat")
    matlab = loadmat(matlab)
    tfs_matlab = matlab["tfs"]

    # Cut away some frequencies - the matlab version performs internal
    # interpolation resulting in aliasing. The rest of the values are a very
    # good fit.
    tfs = tfs[200:, :]
    tfs_matlab = tfs_matlab[200:, :]

    # Some tolerance is needed to due numeric differences.
    tolerance = 1E-5
    min_value = np.abs(tfs).max() * tolerance
    tfs[np.abs(tfs) < min_value] = 0 + 0j
    tfs_matlab[np.abs(tfs_matlab) < min_value] = 0 + 0j

    np.testing.assert_allclose(np.abs(tfs), np.abs(tfs_matlab))
    np.testing.assert_allclose(np.angle(tfs), np.angle(tfs_matlab))


def test_adjoint_time_frequency_phase_misfit_source_plot(tmpdir):
    """
    Tests the plot for a time-frequency misfit adjoint source.
    """
    obs, syn = obspy.read(os.path.join(data_dir, "adj_src_test.mseed")).traces

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))

    ad_src_tf_phase_misfit.adsrc_tf_phase_misfit(obs.times(), obs.data,
                                                 syn.data, 20.0, 100.0,
                                                 plot=True)
    # High tolerance as fonts are for some reason shifted on some systems.
    # This should still be safe as a differnce in the actual tf difference
    # or the waveforms would induce changes all over the plot which would
    # make the rms error much larger.
    images_are_identical("tf_adjoint_source", str(tmpdir), tol=30)


def test_time_frequency_adjoint_source():
    """
    Test the time frequency misfit and adjoint source.
    """
    obs, syn = obspy.read(os.path.join(data_dir, "adj_src_test.mseed")).traces
    ret_val = ad_src_tf_phase_misfit.adsrc_tf_phase_misfit(
        obs.times(), obs.data, syn.data, 20.0, 100.0)

    assert round(ret_val["misfit_value"], 4) == 0.7147
    assert not ret_val["details"]["messages"]

    adj_src_baseline = np.load(os.path.join(
        data_dir, "adjoint_source_baseline.npy"))

    np.testing.assert_allclose(
        actual=ret_val["adjoint_source"],
        desired=adj_src_baseline,
        atol=1E-5 * abs(adj_src_baseline).max(),
        rtol=1E-5)
