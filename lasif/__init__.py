#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Attempt to make sure the number of OpenBLAS threads is correct.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"


class LASIFException(Exception):
    """
    Base exception class for LASIF.
    """
    pass