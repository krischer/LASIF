#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LASIF (LArge-scale Seismic Inversion Framework)

Data management for seismological full seismic waveform inversions using the
Salvus suite of tools.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de) and
    Andreas Fichtner (A.Fichtner@uu.nl) 2012-2017
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
    version="2.0.0a",
    description="",
    author="Lion Krischer and Andreas Fichtner",
    author_email="krischer@geophysik.uni-muenchen.de",
    url="https://github.com/krischer/LASIF",
    packages=find_packages(),
    license="GNU General Public License, version 3 (GPLv3)",
    platforms="OS Independent",
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'],
    install_requires=[
        "obspy>=1.0.3",
        "pyasdf",
        "toml",
        "geographiclib",
        "colorama",
        "mpi4py",
        "numexpr",
        "pytest",
        "flask",
        "flask-cache",
        "geojson"],
    package_data={
        "lasif": get_package_data()},
    entry_points={
        "console_scripts": [
            "lasif = lasif.scripts.lasif_cli:main",
            "iris2quakeml = lasif.scripts.iris2quakeml:main"
        ]
    }
)


if __name__ == "__main__":
    setup(**setup_config)
