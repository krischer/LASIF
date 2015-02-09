#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functionality for data preprocessing.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import time

from lasif.tools.colored_logger import ColoredLogger
from lasif.tools.parallel_helpers import parallel_map


def launch_processing(data_generator, processing_function,
                      iteration, log_filename=None, waiting_time=1.0,
                      process_params=None):
    """
    Launch the parallel processing.

    :param data_generator: A generator yielding file information as required.
    :param log_filename: If given, a log will be written to that file.
    :param waiting_time: The time spent sleeping after the initial message has
        been printed. Useful if the user should be given the chance to cancel
        the processing.
    :param process_params: If given, the processing parameters will be written
        to the logfile.
    """
    logger = ColoredLogger(log_filename=log_filename)

    logger.info("Launching preprocessing using all processes...\n"
                "This might take a while. Press Ctrl + C to cancel.\n")

    # Give the user some time to read the message.
    time.sleep(waiting_time)
    results = parallel_map(
        processing_function,
        ({"processing_info": i, "iteration": iteration} for i in
         data_generator),
        verbose=50, pre_dispatch="all")

    # Keep track of all files.
    successful_file_count = 0
    warning_file_count = 0
    failed_file_count = 0
    total_file_count = len(results)

    for result in results:
        if result.exception is not None:
            filename = result.func_args["processing_info"]["input_filename"]
            msg = "Exception processing file '%s'. %s\n%s" % (
                filename, result.exception, result.traceback)
            logger.error(msg)
            failed_file_count += 1
        elif result.warnings:
            warning_file_count += 1
        else:
            successful_file_count += 1

    return {
        "failed_file_count": failed_file_count,
        "warning_file_count": warning_file_count,
        "total_file_count": total_file_count,
        "successful_file_count": successful_file_count}
