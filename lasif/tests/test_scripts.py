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
        "Extent in latitudinal (X) direction: 3339.6 km, 31 km/element",
        "Extent in longitudinal (Y) direction: 5232.0 km, 31 km/element",
        "Extent in depth (Z) direction: 1171.0 km , 18.3 km/element",
        "",
        "P wave velocities range from 5.8 km/s to 11.7 km/s. The velocities "
        "of the top 5 km",
        "have not been analyzed to avoid very slow layers.",
        "",
        "Maximal recommended time step: 0.080 s",
        "Minimal resolvable period: 10.7 s",
        "",
        "Possible recommended domain decompositions:",
        "Total CPU count:   864; CPUs in X:   9 (12 elements/core), CPUs in "
        "Y:  12 (14 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  1008; CPUs in X:   9 (12 elements/core), CPUs in "
        "Y:  14 (12 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  1152; CPUs in X:  12 ( 9 elements/core), CPUs in "
        "Y:  12 (14 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  1344; CPUs in X:  12 ( 9 elements/core), CPUs in "
        "Y:  14 (12 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  1512; CPUs in X:   9 (12 elements/core), CPUs in "
        "Y:  21 ( 8 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  1728; CPUs in X:   9 (12 elements/core), CPUs in "
        "Y:  24 ( 7 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  1728; CPUs in X:  18 ( 6 elements/core), CPUs in "
        "Y:  12 (14 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  1728; CPUs in X:   9 (12 elements/core), CPUs in "
        "Y:  12 (14 elements/core), CPUs in Z:  16 ( 4 elements/core)",
        "Total CPU count:  2016; CPUs in X:  12 ( 9 elements/core), CPUs in "
        "Y:  21 ( 8 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  2016; CPUs in X:   9 (12 elements/core), CPUs in "
        "Y:  28 ( 6 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  2016; CPUs in X:  18 ( 6 elements/core), CPUs in "
        "Y:  14 (12 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  2016; CPUs in X:   9 (12 elements/core), CPUs in "
        "Y:  14 (12 elements/core), CPUs in Z:  16 ( 4 elements/core)",
        "Total CPU count:  2304; CPUs in X:  12 ( 9 elements/core), CPUs in "
        "Y:  24 ( 7 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  2304; CPUs in X:  12 ( 9 elements/core), CPUs in "
        "Y:  12 (14 elements/core), CPUs in Z:  16 ( 4 elements/core)",
        "Total CPU count:  2592; CPUs in X:  27 ( 4 elements/core), CPUs in "
        "Y:  12 (14 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  2688; CPUs in X:  12 ( 9 elements/core), CPUs in "
        "Y:  28 ( 6 elements/core), CPUs in Z:   8 ( 8 elements/core)",
        "Total CPU count:  2688; CPUs in X:  12 ( 9 elements/core), CPUs in "
        "Y:  14 (12 elements/core), CPUs in Z:  16 ( 4 elements/core)"
    ]

    for expected_line, line in zip(lines, out.splitlines()):
        line = line.strip()
        assert expected_line == line
