#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for some scripts shipping with LASIF.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import re
import os


def test_ses3d_helper_script(capfd):
    """
    Simple test of the helper script.
    """
    os.system("python -m lasif.scripts.ses3d_setup_helper 30 47 1171 108 168 "
              "64")
    out, err = capfd.readouterr()
    assert err == ""
    lines = [
        "SES3D Setup Assistant",
        "",
        "All calculations are done quick and dirty so take them with a grain "
        "of salt.",
        "",
        "Possible recommended domain decompositions:",
        "",
        "Total CPU count:  1344; CPUs in X:  12 ( 9 elements/core), "
        "CPUs in Y:  14 (12 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Extent in latitudinal  (X) direction:  3339.6 km,  27.8 km/element,  "
        "120 elements",
        "Extent in longitudinal (Y) direction:  5232.0 km,  28.7 km/element,  "
        "182 elements",
        "Extent in depth        (Z) direction:  1171.0 km,  16.3 km/element,  "
        " 72 elements",
        "Wave velocities range from 3.9 km/s to 11.7 km/s. The velocities of"
        " the top 15 km have not been analyzed to avoid very slow layers.",
        "Maximal recommended time step: 0.071 s",
        "Minimal resolvable period: 11.8 s",
        "SES3D Settings: nx_global: 108, ny_global: 168, nz_global: 64",
        "px: 12, py: 14, px: 8",
        "",
        "Total CPU count:  2016; CPUs in X:  12 ( 9 elements/core), "
        "CPUs in Y:  21 ( 8 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Extent in latitudinal  (X) direction:  3339.6 km,  27.8 km/element,  "
        "120 elements",
        "Extent in longitudinal (Y) direction:  5232.0 km,  27.7 km/element,  "
        "189 elements",
        "Extent in depth        (Z) direction:  1171.0 km,  16.3 km/element,  "
        " 72 elements",
        "Wave velocities range from 3.9 km/s to 11.7 km/s. The velocities "
        "of the top 15 km have not been analyzed to avoid very slow layers.",
        "Maximal recommended time step: 0.071 s",
        "Minimal resolvable period: 11.4 s",
        "SES3D Settings: nx_global: 108, ny_global: 168, nz_global: 64",
        "px: 12, py: 21, px: 8",
        "",
        "Total CPU count:  2304; CPUs in X:  12 ( 9 elements/core), "
        "CPUs in Y:  24 ( 7 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Extent in latitudinal  (X) direction:  3339.6 km,  27.8 km/element,  "
        "120 elements",
        "Extent in longitudinal (Y) direction:  5232.0 km,  27.3 km/element,  "
        "192 elements",
        "Extent in depth        (Z) direction:  1171.0 km,  16.3 km/element,  "
        " 72 elements",
        "Wave velocities range from 3.9 km/s to 11.7 km/s. The velocities "
        "of the top 15 km have not been analyzed to avoid very slow layers.",
        "Maximal recommended time step: 0.071 s",
        "Minimal resolvable period: 11.4 s",
        "SES3D Settings: nx_global: 108, ny_global: 168, nz_global: 64",
        "px: 12, py: 24, px: 8"]

    # Remove color codes from output.
    color_codes = re.compile(r"\033\[\d+(?:;\d+)?m")

    for expected_line, line in zip(lines, out.splitlines()):
        line = color_codes.sub("", line).strip()
        assert expected_line == line
