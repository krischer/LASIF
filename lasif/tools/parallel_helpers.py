#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helpers for embarrassingly parallel calculations using MPI. All functions
works just fine when running on one core and not started with MPI.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014-2015
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import collections
import colorama
import functools
import inspect
import itertools
import os
import sys
import traceback
import warnings

from mpi4py import MPI


class FunctionInfo(collections.namedtuple(
    "FunctionInfo", ["func_args", "result", "warnings", "exception",
                     "traceback"])):
    """
    Namedtuple used to collect information about a function execution.

    It has the following fields: ``func_args``, ``result``, ``warnings``,
    ``exception``, and ``traceback``.
    """
    pass


def function_info(traceback_limit=10):
    """
    Decorator collecting information during the execution of a function.

    This is useful for collecting information about a function runnning on a
    number of processes/machines.

    It returns a FunctionInfo named tuple with the following fields:

    * ``func_args``: Dictionary containing all the functions arguments and
      values.
    * ``result``: The return value of the function. Will be None if an
      exception has been raised.
    * ``warnings``: A list with all warnings the function raised.
    * ``exception``: The exception the function raised. Will be None, if no
      exception has been raised.
    * ``traceback``: The full traceback in case an exception occured as a
      string.
      A traceback object is not serializable thus a string is used.


    >>> @function_info()
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
    def _function_info(f):
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
                    stack = traceback.extract_stack(limit=traceback_limit)
                    tb = traceback.extract_tb(exc_info[2])
                    full_tb = stack[:-1] + tb
                    exc_line = traceback.format_exception_only(*exc_info[:2])
                    tb = "Traceback (%i levels - most recent call last):\n" % \
                        traceback_limit
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
    return _function_info


def _execute_wrapped_function(func, parameters):
    """
    Helper function to execute the same function but wrapper with the
    function_info decorator.

    This is necessary as a function needs to be importable, otherwise pickle
    does not work with it.
    """
    return function_info()(func)(**parameters)


def distribute_across_ranks(function, items, get_name, logfile):
    """
    Calls a function once for each item.

    It will be distributed across MPI ranks if launched with MPI.

    :param function: The function to be executed for each item.
    :param items: The function will be executed once for each item. It
        expects a list of dictionaries so that ``function(**item)`` can work.
        Only rank 0 needs to pass this. It will be ignored coming from other
        ranks.
    :param get_name: Function to extract a name for each item to be able to
        produce better logfiles.
    :param logfile: The logfile to write.
    """
    def split(container, count):
        """
        Simple and elegant function splitting a container into count
        equal chunks.

        Order is not preserved but for the use case at hand this is
        potentially an advantage as data sitting in the same folder thus
        have a higher at being processed at the same time thus the disc
        head does not have to jump around so much. Of course very
        architecture dependent.
        """
        return [container[_i::count] for _i in range(count)]

    # Rank zero collects what needs to be done and distributes it across
    # all cores.
    if MPI.COMM_WORLD.rank == 0:
        total_length = len(items)
        items = split(items, MPI.COMM_WORLD.size)
    else:
        items = None

    # Now each rank knows what it has to process. This still works
    # nicely with only one core, the overhead is negligible.
    items = MPI.COMM_WORLD.scatter(items, root=0)

    results = []

    for _i, item in enumerate(items):
        results.append(_execute_wrapped_function(function, item))

        if MPI.COMM_WORLD.rank == 0:
            print("Approximately %i of %i items have been processed." % (
                min((_i + 1) * MPI.COMM_WORLD.size, total_length),
                total_length))

    results = MPI.COMM_WORLD.gather(results, root=0)

    if MPI.COMM_WORLD.rank != 0:
        return

    results = list(itertools.chain.from_iterable(results))

    successful_file_count = 0
    warning_file_count = 0
    failed_file_count = 0
    total_file_count = len(results)

    # Log the results.
    with open(logfile, "wt") as fh:
        for result in results:
            fh.write("\n============\nItem: %s" % get_name(result.func_args))
            has_exception = False
            has_warning = False
            if result.exception:
                has_exception = True
                fh.write("\n")
                fh.write(result.traceback)
            elif result.warnings:
                has_warning = True
                for w in result.warnings:
                    fh.write("\nWarning: %s\n" % str(w))
            else:
                fh.write(" - SUCCESS")

            if has_exception:
                failed_file_count += 1
            elif has_warning:
                warning_file_count += 1
            else:
                successful_file_count += 1

        print("\nFinished processing %i items. See the logfile for "
              "details.\n" % total_file_count)
        print("\t%s%i files failed being processed.%s" %
              (colorama.Fore.RED, failed_file_count,
               colorama.Fore.RESET))
        print("\t%s%i files raised warnings while being processed.%s" %
              (colorama.Fore.YELLOW, warning_file_count,
               colorama.Fore.RESET))
        print("\t%s%i files have been processed without errors or warnings%s" %
              (colorama.Fore.GREEN, successful_file_count,
               colorama.Fore.RESET))

        print("\nLogfile written to '%s'." % os.path.relpath(logfile))

    return results
