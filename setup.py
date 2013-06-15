#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASIF (LArge-scale Seismic Inversion Framework)

A collection of scripts that are useful for running a full waveform inversion
workflow with SES3D 4.0.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de) and
    Andreas Fichtner (A.Fichtner@uu.nl) 2012-2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from setuptools import setup

setup_config = dict(
    name="lasif",
    version="0.0.1a",
    description="",
    author="Lion Krischer and Andreas Fichtner",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="",
    packages=["lasif"],
    package_dir={"lasif": "lasif"},
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
    install_requires=["obspy >= 0.8.3", "progressbar", "requests",
        "geographiclib", "numpy", "matplotlib", "lxml"],
    entry_points={
        # Register the console scripts.
        "console_scripts": [
            "lasif = lasif.scripts.lasif_cli:main",
            "iris2quakeml = lasif.scripts.iris2quakeml:main"
        ],
        # Register the SES3D reading function with ObsPy.
        "obspy.plugin.waveform": "SES3D = lasif.ses3d_file_parser",
        "obspy.plugin.waveform.SES3D": [
            "isFormat = lasif.ses3d_file_parser:is_SES3D",
            "readFormat = lasif.ses3d_file_parser:read_SES3D"
        ]
    }
)

if __name__ == "__main__":
    setup(**setup_config)
