#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Attempt to make sure the number of OpenBLAS threads is correct.
import os
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


class LASIFWarning(UserWarning):
    """
    Base warning class for LASIF.
    """
    pass