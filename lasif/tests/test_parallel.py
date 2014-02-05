#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for the parallel helper methods.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
from lasif.tools.parallel import function_info
import warnings


def test_function_info_decorator():
    """
    Test suite for the function info wrapper.
    """
    @function_info
    def test_function(a, b=2):
        return a / b

    # Normal behaviour.
    info = test_function(4, 4)
    assert info.func_args == {"a": 4, "b": 4}
    assert info.result == 1
    assert info.warnings == []
    assert info.exception is None
    assert info.traceback is None

    # Check the difference between args and kwargs.
    info = test_function(4)
    assert info.func_args == {"a": 4, "b": 2}
    assert info.result == 2
    assert info.warnings == []
    assert info.exception is None
    assert info.traceback is None

    # Division with zero should raise an exception.
    info = test_function(4, 0)
    assert info.func_args == {"a": 4, "b": 0}
    assert info.result is None
    assert info.warnings == []
    assert type(info.exception) is ZeroDivisionError
    assert "ZeroDivisionError" in info.traceback

    @function_info
    def function_triggering_some_warnings():
        warnings.warn("First Warning", SyntaxWarning)
        warnings.warn("Second Warning", UserWarning)
        return 0

    info = function_triggering_some_warnings()
    assert info.func_args == {}
    assert info.result == 0
    assert info.exception is None
    assert info.traceback is None
    assert len(info.warnings) == 2
    assert info.warnings[0].category is SyntaxWarning
    assert info.warnings[1].category is UserWarning
    assert str(info.warnings[0].message) == "First Warning"
    assert str(info.warnings[1].message) == "Second Warning"
