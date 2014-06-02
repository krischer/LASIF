#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import io
import obspy
import os
import pytest
import time
import re

from lasif import LASIFNotFoundError, LASIFWarning
from lasif.components.waveforms import WaveformsComponent
from lasif.components.communicator import Communicator


@pytest.fixture
def comm():
    """
    Returns a communicator with an initialized events component.
    """
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "ExampleProject")
    data_folder = os.path.join(proj_dir, "DATA")
    synthetics_folder = os.path.join(proj_dir, "SYNTHETICS")
    comm = Communicator()
    WaveformsComponent(data_folder, synthetics_folder, comm, "waveforms")
    return comm


def test_get_raw_metadata(comm):
    event_name = "GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11"
    print comm.waveforms.get_metadata_raw(event_name)


