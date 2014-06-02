#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file has a two purpose:

1. Force the use of the "Agg" matplotlib backend for the tests. Otherwise lots
  of windows pop up.
2. Force OpenBLAS to use single threading. Otherwise it will deadlock in some
  cases.

It will be run by pytest, as specified in the pytest.ini file.
"""
# Make sure this is executed before any test. Otherwise the import order is not
# really clear.
# This is also specified in all __init__.py's
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.switch_backend("agg")

# Fixtures will be available in the whole module.
from lasif.tests.components.test_stations_component import comm as \
    stations_comm
from lasif.tests.components.test_events_component import comm as \
    events_comm
from lasif.tests.components.test_inventory_db_component import filled_comm as \
    inventory_db_comm
from lasif.tests.components.test_waveforms_component import comm as \
    waveforms_comm
from lasif.tests.components.test_iterations_component import comm as \
    iterations_comm
from lasif.tests.components.test_query_component import comm as \
    query_comm


def pytest_runtest_setup(item):
    """
    This hook is called before every test.
    """
    plt.switch_backend("agg")
    assert matplotlib.get_backend().lower() == "agg"
