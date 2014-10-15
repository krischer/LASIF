#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Attempt to make sure the number of OpenBLAS threads is correct.
import inspect
import os
from subprocess import Popen, PIPE
import warnings

os.environ["OPENBLAS_NUM_THREADS"] = "1"


class LASIFError(Exception):
    """
    Base exception class for LASIF.
    """
    pass


class LASIFNotFoundError(LASIFError):
    """
    Raised whenever something is not found inside the project.
    """
    pass


class LASIFAdjointSourceCalculationError(LASIFError):
    """
    Raised when something goes wrong when calculating an adjoint source.
    """
    pass


class LASIFWarning(UserWarning):
    """
    Base warning class for LASIF.
    """
    pass


# Determine the version of LASIF. Using git.
__root_path = os.path.abspath(os.path.dirname(os.path.dirname(inspect.getfile(
    inspect.currentframe()))))
try:
    p = Popen(['git', 'describe', '--dirty', '--abbrev=4', '--always',
               '--tags'], cwd=__root_path, stdout=PIPE, stderr=PIPE)
    p.stderr.close()
    line = p.stdout.readline().decode()
    p.stdout.close()
    __version__ = line.strip()
except:
    warnings.warn("Could not determine LASIF version. Is git installed?",
                  LASIFWarning)
    __version__ = "UNDEFINED"
