#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests all Python files of the project with flake8. This ensure PEP8 conformance
and some other sanity checks as well.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import flake8
import flake8.main
import inspect
import os
import warnings


def test_flake8():
    if flake8.__version__ <= "2":
        msg = ("Module was designed to be tested with flake8 >= 2.0. "
               "Please update.")
        warnings.warn(msg)
    test_dir = os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe())))
    lasif_dir = os.path.dirname(test_dir)
    error_count = 0
    file_count = 0
    for dirpath, _, filenames in os.walk(lasif_dir):
        filenames = [_i for _i in filenames if
                     os.path.splitext(_i)[-1] == os.path.extsep + "py"]
        if not filenames:
            continue
        for py_file in filenames:
            file_count += 1
            full_path = os.path.join(dirpath, py_file)
            if flake8.main.check_file(full_path):
                error_count += 1
    assert file_count > 10
    assert not error_count
