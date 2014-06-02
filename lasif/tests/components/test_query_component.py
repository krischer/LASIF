#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import inspect
import os
import pytest
import shutil

from lasif.components.communicator import Communicator
from lasif.components.events import EventsComponent
from lasif.components.inventory_db import InventoryDBComponent
from lasif.components.query import QueryComponent
from lasif.components.stations import StationsComponent
from lasif.components.waveforms import WaveformsComponent


@pytest.fixture()
def comm(tmpdir):
    proj_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))), "data", "ExampleProject")
    tmpdir = str(tmpdir)
    shutil.copytree(proj_dir, os.path.join(tmpdir, "proj"))
    proj_dir = os.path.join(tmpdir, "proj")

    comm = Communicator()

    # Init the Inventory DB component.
    db_file = os.path.join(tmpdir, "inventory_db.sqlite")
    InventoryDBComponent(
        db_file=db_file,
        communicator=comm,
        component_name="inventory_db")

    # Init the waveform component.
    data_folder = os.path.join(proj_dir, "DATA")
    synthetics_folder = os.path.join(proj_dir, "SYNTHETICS")
    WaveformsComponent(data_folder, synthetics_folder, comm, "waveforms")

    # Init the events component.
    EventsComponent(os.path.join(proj_dir, "EVENTS"), comm, "events")

    # Init the stations component.
    StationsComponent(
        stationxml_folder=os.path.join(proj_dir, "STATIONS",
                                       "StationXML"),
        seed_folder=os.path.join(proj_dir, "STATIONS", "SEED"),
        resp_folder=os.path.join(proj_dir, "STATIONS", "RESP"),
        cache_folder=tmpdir,
        communicator=comm,
        component_name="stations")

    # Finally init the query component.
    QueryComponent(communicator=comm, component_name="query")

    return comm
