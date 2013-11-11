#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple colorful logging helper that prints to a file and the screen.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import colorama
from datetime import datetime
import logging


class ColoredLogger(object):
    """
    Simple logging class printing to the screen in color as well as to a file.
    """
    def __init__(self, log_filename=None, debug=False):
        FORMAT = "[%(asctime)-15s] %(levelname)s: %(message)s"
        if log_filename is not None:
            logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                                format=FORMAT)
        else:
            logging.basicConfig(level=logging.DEBUG, format=FORMAT)
        self.has_file = bool(log_filename)
        self.logger = logging.getLogger("LASIF")
        self.set_debug(debug)

    def set_debug(self, value):
        if value:
            self._debug = True
            self.logger.setLevel(logging.DEBUG)
        else:
            self._debug = False
            self.logger.setLevel(logging.INFO)

    def critical(self, msg):
        print(colorama.Fore.WHITE + colorama.Back.RED +
              self._format_message("CRITICAL", msg) + colorama.Style.RESET_ALL)
        if not self.has_file:
            return
        self.logger.critical(msg)

    def exception(self, msg):
        print(colorama.Fore.WHITE + colorama.Back.RED +
              self._format_message("EXCEPTION", msg) +
              colorama.Style.RESET_ALL)
        if not self.has_file:
            return
        self.logger.exception(msg)

    def error(self, msg):
        print(colorama.Fore.RED + self._format_message("ERROR", msg) +
              colorama.Style.RESET_ALL)
        if not self.has_file:
            return
        self.logger.error(msg)

    def warning(self, msg):
        print(colorama.Fore.YELLOW + self._format_message("WARNING", msg) +
              colorama.Style.RESET_ALL)
        if not self.has_file:
            return
        self.logger.warning(msg)

    def info(self, msg):
        print(self._format_message("INFO", msg))
        if not self.has_file:
            return
        self.logger.info(msg)

    def debug(self, msg):
        if not self._debug:
            return
        print(colorama.Fore.BLUE + self._format_message("DEBUG", msg) +
              colorama.Style.RESET_ALL)
        if not self.has_file:
            return
        self.logger.debug(msg)

    def _format_message(self, prefix, msg):
        return "[%s] %s: %s" % (datetime.now(), prefix, msg)
