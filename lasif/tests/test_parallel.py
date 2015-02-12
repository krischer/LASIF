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
import os
import warnings

from lasif.tools.parallel_helpers import function_info, distribute_across_ranks


def test_function_info_decorator():
    """
    Test suite for the function info wrapper.
    """
    @function_info()
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

    @function_info()
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


def test_traceback_limiting_for_function_info_decorator():
    """
    Tests that the traceback depth limiting works for the function info
    decorator.
    """
    # Defaults to three.
    @function_info()
    def test_function(a):
        return 1 / a

    info = test_function(0)
    tb_default_length = len(info.traceback)
    assert tb_default_length > 0

    # Test for a limit of 2.
    @function_info(traceback_limit=2)
    def test_function(a):
        return 1 / a

    info = test_function(0)
    tb_length_limit_2 = len(info.traceback)
    assert tb_length_limit_2 < tb_default_length

    # Test for a limit of 1
    @function_info(traceback_limit=1)
    def test_function(a):
        return 1 / a

    info = test_function(0)
    tb_length_limit_1 = len(info.traceback)
    assert tb_length_limit_1 < tb_length_limit_2


def __random_fct(a, b, c=0):
    """
    Helper function.
    """
    if c == 1:
        warnings.warn("First Warning", SyntaxWarning)
        warnings.warn("Second Warning", UserWarning)
    return a / b


def test_distribute_across_ranks(tmpdir):
    """
    Test the distribute across ranks method at least a bit. This test
    naturally only runs without MPI.
    """
    def input_generator():
        yield {"a": 2, "b": 1}  # results in 2
        yield {"a": 4, "b": 0}  # results in None, an exception,
        # and a traceback.
        yield {"a": 1, "b": 1, "c": 1}  # results in 1 and two warnings.
        raise StopIteration

    logfile = os.path.join(str(tmpdir), "log.txt")

    results = distribute_across_ranks(
        function=__random_fct, items=list(input_generator()),
        get_name=lambda x: str(x), logfile=logfile)

    assert os.path.exists(logfile)

    # Sort them with the expected result to be able to compare them. The order
    # is not guaranteed when using multiple processes.
    results.sort(key=lambda x: x.result)

    assert results[0].result is None
    assert results[0].func_args == {"a": 4, "b": 0, "c": 0}
    assert results[0].warnings == []
    assert type(results[0].exception) is ZeroDivisionError
    assert "ZeroDivisionError" in results[0].traceback

    assert results[1].result == 1
    assert results[1].func_args == {"a": 1, "b": 1, "c": 1}
    assert results[1].exception is None
    assert results[1].traceback is None
    assert len(results[1].warnings) == 2
    assert results[1].warnings[0].category is SyntaxWarning
    assert results[1].warnings[1].category is UserWarning
    assert str(results[1].warnings[0].message) == "First Warning"
    assert str(results[1].warnings[1].message) == "Second Warning"

    assert results[2].result == 2
    assert results[2].func_args == {"a": 2, "b": 1, "c": 0}
    assert results[2].warnings == []
    assert results[2].exception is None
    assert results[2].traceback is None
