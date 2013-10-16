#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file has a single purpose: Force the use of the "Agg" matplotlib backend
for the tests. Otherwise lots of windows pop up.

It will be run by pytest, as specified in the pytest.ini file.
"""
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.switch_backend("agg")


def pytest_runtest_setup(item):
    """
    This hook is called before every test.
    """
    plt.switch_backend("agg")
    assert matplotlib.get_backend().lower() == "agg"
