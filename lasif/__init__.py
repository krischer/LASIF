#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .version import get_git_version

__version__ = get_git_version()


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
