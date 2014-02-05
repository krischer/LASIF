#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helpers for embarissingly parallel calculations.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import collections
import pp

# Use multiprocessing only to derive the number of available processors.
from multiprocessing import cpu_count
cpu_count = cpu_count()


FunctionInfo = collections.namedtuple(
    "FunctionInfo", ["func_args", "result", "warnings", "exception",
                     "traceback"])


def function_info(f):
    """
    Decorator returning information about the function.

    :param function:
    :param arguments:
    :return:
    """
    import functools
    import inspect
    import sys
    import traceback
    import warnings

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


def parallel(function, iterable):
    """
    Custom unordered map implementation using pp's processes.

    This enables arbitrarily long queues.

    :param function: The function to run for each item.
    :param iterable: The iterable yielding items.
    :param processes: The number of processes to launch.
    """

    def yield_job(job):
        _catch_output = StringIO()
        old_stdout = sys.stdout
        sys.stdout = _catch_output

        job_warnings = job()

        sys.stdout = old_stdout
        _catch_output.tell()
        exception = _catch_output.read()

        if not exception:
            exception = None

        return {
            "result": job_result,
            "arguments": job_arguments,
            "exception": job_exception,
            "warnings": job_warnings
        }

    job_server = pp.Server()
    num_jobs = job_server.get_ncpus()
    active_jobs = []

    while True:
        # Give some time.
        time.sleep(0.1)

        try:
            value = iterable.next()
        except StopIteration:
            break

        # Collect any finished jobs.
        for job in active_jobs[:]:
            if job.finished is True:
                yield yield_job(job)
                active_jobs.remove(job)

        # Start new jobs. Buffer a certain number of jobs.
        jobs_left = True
        while len(active_jobs) < int(num_jobs * 1.5):
            # Get new values until there are no left. Then break.
            try:
                value = iterable.next()
            except StopIteration:
                jobs_left = False
                break
            new_job = job_server.submit(function, (value, ))
            new_job.filename = value["data_path"]
            active_jobs.append(new_job)

        # End job submission and collect remaining jobs.
        if jobs_left is False:
            for job in active_jobs:
                yield yield_job(job)
            break
