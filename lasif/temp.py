#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporary code for LASIF.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import numpy as np
from lasif.adjoint_sources.ad_src_tf_phase_misfit import adsrc_tf_phase_misfit


def adjoint_src_for_window(data, synth, starttime, endtime, weight,
                           process_parameters, output_window_manager,
                           output_adjoint_source_manager):

    output_window_manager.write_window(
        data.id, starttime, endtime, weight, "cosine",
        "TimeFrequencyPhaseMisfitFichtner2008")

    #  Decimal percentage of cosine taper (ranging from 0 to 1). Set to the
    # fraction of the minimum period to the window length.
    taper_percentage = np.min(
        [1.0, 1.0 / process_parameters["lowpass"] / (endtime - starttime)])

    #  data
    data_trimmed = data.copy()
    data_trimmed.trim(starttime, endtime)
    data_trimmed.taper(type='cosine',
                       max_percentage=0.5 * taper_percentage)
    data_trimmed.trim(synth.stats.starttime, synth.stats.endtime, pad=True,
                      fill_value=0.0)

    #  synthetics
    synth_trimmed = synth.copy()
    synth_trimmed.trim(starttime, endtime)
    synth_trimmed.taper(type='cosine',
                        max_percentage=0.5 * taper_percentage)
    synth_trimmed.trim(synth.stats.starttime, synth.stats.endtime,
                       pad=True, fill_value=0.0)

    #  make time axis
    t = np.linspace(0, synth.stats.npts * synth.stats.delta,
                    synth.stats.npts)

    #  clear axes of misfit plot ------------------------------------------

    #  set data and synthetics, compute actual misfit ---------------------

    t = np.require(t, dtype="float64", requirements="C")
    data_d = np.require(data_trimmed.data, dtype="float64",
                        requirements="C")
    synth_d = np.require(synth_trimmed.data, dtype="float64",
                         requirements="C")

    #  compute misfit and adjoint source
    adsrc = adsrc_tf_phase_misfit(
        t, data_d, synth_d,
        1.0 / process_parameters["lowpass"],
        1.0 / process_parameters["highpass"],
        axis=None, colorbar_axis=None)

    #  write adjoint source to file ---------------------------------------
    output_adjoint_source_manager.write_adjoint_src(
        adsrc["adjoint_source"], data.id, starttime, endtime)
