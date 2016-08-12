#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASIF (LArge-scale Seismic Inversion Framework)

A collection of scripts that are useful for running a full waveform inversion
workflow with SES3D 4.0.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de) and
    Andreas Fichtner (A.Fichtner@uu.nl) 2012-2016
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import inspect
import os
from setuptools import setup, find_packages


def get_package_data():
    """
    Returns a list of all files needed for the installation relativ to the
    "lasif" subfolder.
    """
    filenames = []
    # The lasif root dir.
    root_dir = os.path.join(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe()))), "lasif")
    # Recursively include all files in these folders:
    folders = [os.path.join(root_dir, "tests", "baseline_images"),
               os.path.join(root_dir, "tests", "data")]
    for folder in folders:
        for directory, _, files in os.walk(folder):
            for filename in files:
                # Exclude hidden files.
                if filename.startswith("."):
                    continue
                filenames.append(os.path.relpath(
                    os.path.join(directory, filename),
                    root_dir))
    return filenames


setup_config = dict(
    name="lasif",
    version="0.1.x",
    description="",
    author="Lion Krischer and Andreas Fichtner",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="https://github.com/krischer/LASIF",
    packages=find_packages(),
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
    install_requires=[
        "obspy>=1.0.1",
        "progressbar",
        "geographiclib",
        "numpy",
        "colorama",
        "matplotlib",
        "mpi4py",
        "joblib",
        "lxml",
        "nose",
        "numexpr",
        "pytest",
        "mock",
        "flask",
        "flask-cache",
        "geojson"],
    package_data={
        "lasif": get_package_data()},
    entry_points={
        # Register the console scripts.
        "console_scripts": [
            "lasif = lasif.scripts.lasif_cli:main",
            "iris2quakeml = lasif.scripts.iris2quakeml:main"
        ],
        # Register the SES3D reading function with ObsPy.
        "obspy.plugin.waveform":
            "SES3D = lasif.file_handling.s3d_file_parser",
        "obspy.plugin.waveform.SES3D": [
            "isFormat = lasif.file_handling.ses3d_file_parser:is_SES3D",
            "readFormat = lasif.file_handling.ses3d_file_parser:read_SES3D"
        ]
    }
)


if __name__ == "__main__":
    setup(**setup_config)
