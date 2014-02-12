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
from lasif.tools.parallel import function_info, parallel_map
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


def __random_fct(a, b, c=0):
    """
    Helper function as functions need to be importable for multiprocessing to
    work.
    """
    if c == 1:
        warnings.warn("First Warning", SyntaxWarning)
        warnings.warn("Second Warning", UserWarning)
    return a / b


def test_parallel_map():
    """
    Test the parallel mapping method.
    """
    def input_generator():
        yield {"a": 2, "b": 1}  # results in 2
        yield {"a": 4, "b": 0}  # results in None, an exception,
                                # and a traceback.
        yield {"a": 1, "b": 1, "c": 1}  # results in 1 and two warnings.
        raise StopIteration

    results = parallel_map(__random_fct, input_generator())

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


def __random_fct_2(*args):
    """
    Helper function as functions need to be importable for multiprocessing to
    work.
    """
    import obspy
    st = obspy.read()
    st += st.copy()
    # This triggers a call to numpy.linalg and thus a BLAS routine.
    st.detrend("linear")


def test_BLAS():
    """
    Simple test asserting that it works with BLAS which is problematic for a
    number of reasons.

    If this test returns it gives some confidence that the machine is able to
    run the type of parallel processing implemented in LASIF.
    """
    results = parallel_map(__random_fct_2, [{}] * 4, n_jobs=2)
    assert len(results) == 4
