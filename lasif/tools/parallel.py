#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helpers for embarrassingly parallel calculations.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import collections
import functools
import inspect
import joblib
import numpy as np
import sys
import traceback
import warnings


class FunctionInfo(collections.namedtuple(
    "FunctionInfo", ["func_args", "result", "warnings", "exception",
                     "traceback"])):
    """
    Namedtuple used to collect information about a function execution.

    It has the following fields: ``func_args``, ``result``, ``warnings``,
    ``exception``, and ``traceback``.
    """
    pass


def function_info(f):
    """
    Decorator collecting information during the execution of a function.

    This is useful for collecting information about a function runnning on a
    number of processes/machines.

    It returns a FunctionInfo named tuple with the following fields:

    * ``func_args``: Dictionary containing all the functions arguments and values.
    * ``result``: The return value of the function. Will be None if an exception
      has been raised.
    * ``warnings``: A list with all warnings the function raised.
    * ``exception``: The exception the function raised. Will be None, if no
      exception has been raised.
    * ``traceback``: The full traceback in case an exception occured as a string.
      A traceback object is not serializable thus a string is used.


    >>> @function_info
    ... def test(a, b=2):
    ...     return a / b
    >>> info = test(4, 1)
    >>> info.func_args
    {'a': 4, 'b': 1}
    >>> info.result
    4

    ``warnings`` is empty if no warning has been raised. Otherwise it will
    collect all warnings.

    >>> info.warnings
    []

    ``exception`` and ``traceback`` are ``None`` if the function completed
    successfully.

    >>> info.exception
    >>> info.traceback
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = None
            exception = None
            tb = None
            func_args = inspect.getcallargs(f, *args, **kwargs)
            try:
                result = f(*args, **kwargs)
            except Exception as e:
                # With help from http://stackoverflow.com/a/14528175.
                exc_info = sys.exc_info()
                stack = traceback.extract_stack()
                tb = traceback.extract_tb(exc_info[2])
                full_tb = stack[:-1] + tb
                exc_line = traceback.format_exception_only(*exc_info[:2])
                tb = "Traceback (most recent call last):\n"
                tb += "".join(traceback.format_list(full_tb))
                tb += "\n"
                tb += "".join(exc_line)
                exception = e

        return FunctionInfo(
            func_args=func_args,
            result=result,
            exception=exception,
            warnings=w,
            traceback=tb)

    return wrapper


def __execute_wrapped_function(func, parameters):
    """
    Helper function to execute the same function but wrapper with the
    function_info decorator.

    This is necessary as a function needs to be importable, otherwise pickle
    does not work with it.
    """
    return function_info(func)(**parameters)


def parallel_map(func, iterable, n_jobs=-1, verbose=1,
                 pre_dispatch="1.5*n_jobs"):
    """
    Thin wrapper around joblib.Parallel.

    For one it takes care to use the threading backend if OSX's Accelerate
    Framework is detected. And it also wraps all functions with the
    function_info decorator in order to get a more meaningful output.

    :type n_jobs: int
    :param n_jobs: The number of jobs to use for the computation. If -1 all CPUs
        are used. If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used.
        Thus for n_jobs = -2, all CPUs but one are used.
        Same parameter as in joblib.Parallel.
    :type verbose: int
    :param verbose: The verbosity level: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout. The frequency of the
        messages increases with the verbosity level. If it more than 10, all
        iterations are reported.  Defaults to 1.
        Same parameter as in joblib.Parallel.
    :param pre_dispatch: The amount of jobs to be pre-dispatched. Default is
        ``1.5*n_jobs``.
        Same parameter as in joblib.Parallel.
    """
    backend = "multiprocessing"

    config_info = str([value for key, value in
                       np.__config__.__dict__.iteritems()
                       if key.endswith("_info")]).lower()

    if "accelerate" in config_info or "veclib" in config_info:
        msg = ("NumPy linked against 'Accelerate.framework'. Multiprocessing "
               "will be disabled. See "
               "https://github.com/obspy/obspy/wiki/Notes-on-Parallel-"
               "Processing-with-Python-and-ObsPy for more information.")
        warnings.warn(msg)
        backend = "threading"
        n_jobs = 1


    return joblib.Parallel(n_jobs=n_jobs, verbose=verbose,
                           pre_dispatch=pre_dispatch, backend=backend,
                           )(joblib.delayed(
        __execute_wrapped_function)(func, i) for i in iterable)
