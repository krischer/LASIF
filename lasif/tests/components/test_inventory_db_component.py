#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

from collections import namedtuple
import inspect
import mock
import os
import pytest
import urllib2

from lasif.components.inventory_db import InventoryDBComponent
from lasif.components.communicator import Communicator


@pytest.fixture
def comm(tmpdir):
    tmpdir = str(tmpdir)
    db_file = os.path.join(tmpdir, "inventory_db.sqlite")
    comm = Communicator()
    InventoryDBComponent(
        db_file=db_file,
        communicator=comm,
        component_name="inventory_db")
    return comm


@pytest.fixture
def filled_comm(tmpdir):
    """
    Used for doctests. Needs a separate initialization as for some reason
    doctests do sometimes not like chained fixtures...
    """
    tmpdir = str(tmpdir)
    db_file = os.path.join(tmpdir, "inventory_db.sqlite")
    comm = Communicator()
    InventoryDBComponent(
        db_file=db_file,
        communicator=comm,
        component_name="inventory_db")
    comm.inventory_db.save_station_coordinates(
        station_id="AA.BB", latitude=1.0, longitude=2.0, elevation_in_m=3.0,
        local_depth_in_m=4.0)
    comm.inventory_db.save_station_coordinates(
        station_id="CC.DD", latitude=2.0, longitude=2.0, elevation_in_m=2.0,
        local_depth_in_m=2.0)
    comm.inventory_db.save_station_coordinates(
        station_id="EE.FF", latitude=None, longitude=None, elevation_in_m=None,
        local_depth_in_m=None)
    return comm


def test_save_and_get_station_coordinates(comm):
    comm.inventory_db.save_station_coordinates(
        station_id="AA.BB", latitude=1.0, longitude=2.0, elevation_in_m=3.0,
        local_depth_in_m=4.0)
    comm.inventory_db.save_station_coordinates(
        station_id="CC.DD", latitude=2.0, longitude=2.0, elevation_in_m=2.0,
        local_depth_in_m=2.0)

    assert comm.inventory_db.get_coordinates("AA.BB") == {
        "latitude": 1.0, "longitude": 2.0, "elevation_in_m": 3.0,
        "local_depth_in_m": 4.0}
    assert comm.inventory_db.get_coordinates("CC.DD") == {
        "latitude": 2.0, "longitude": 2.0, "elevation_in_m": 2.0,
        "local_depth_in_m": 2.0}

    assert comm.inventory_db.get_all_coordinates() == {
        "AA.BB": {
            "latitude": 1.0, "longitude": 2.0, "elevation_in_m": 3.0,
            "local_depth_in_m": 4.0},
        "CC.DD": {
            "latitude": 2.0, "longitude": 2.0, "elevation_in_m": 2.0,
            "local_depth_in_m": 2.0}}


def test_dealing_with_coordinate_less_stations(comm):
    comm.inventory_db.save_station_coordinates(
        station_id="AA.BB", latitude=1.0, longitude=2.0, elevation_in_m=3.0,
        local_depth_in_m=4.0)
    # Save an empty station.
    comm.inventory_db.save_station_coordinates(
        station_id="CC.DD", latitude=None, longitude=None,
        elevation_in_m=None, local_depth_in_m=None)

    assert comm.inventory_db.get_coordinates("AA.BB") == {
        "latitude": 1.0, "longitude": 2.0, "elevation_in_m": 3.0,
        "local_depth_in_m": 4.0}
    assert comm.inventory_db.get_coordinates("CC.DD") == {
        "latitude": None, "longitude": None, "elevation_in_m": None,
        "local_depth_in_m": None}

    assert comm.inventory_db.get_all_coordinates() == {
        "AA.BB": {
            "latitude": 1.0, "longitude": 2.0, "elevation_in_m": 3.0,
            "local_depth_in_m": 4.0},
        "CC.DD": {
            "latitude": None, "longitude": None, "elevation_in_m": None,
            "local_depth_in_m": None}}

    # Now remove the one station without coordinates.
    comm.inventory_db.remove_coordinate_less_stations()
    assert comm.inventory_db.get_coordinates("AA.BB") == {
        "latitude": 1.0, "longitude": 2.0, "elevation_in_m": 3.0,
        "local_depth_in_m": 4.0}
    assert comm.inventory_db.get_all_coordinates() == {
        "AA.BB": {
            "latitude": 1.0, "longitude": 2.0, "elevation_in_m": 3.0,
            "local_depth_in_m": 4.0}}


def test_saving_empty_stations(comm):
    """
    Either all coordinate values must be None or only the local depth.
    """
    # Both are fine.
    comm.inventory_db.save_station_coordinates(
        station_id="AA.BB", latitude=1.0, longitude=2.0, elevation_in_m=3.0,
        local_depth_in_m=None)
    comm.inventory_db.save_station_coordinates(
        station_id="CC.DD", latitude=None, longitude=None,
        elevation_in_m=None, local_depth_in_m=None)

    # This one should raise.
    with pytest.raises(ValueError):
        comm.inventory_db.save_station_coordinates(
            station_id="AA.BB", latitude=None, longitude=2.0,
            elevation_in_m=3.0, local_depth_in_m=2.0)

    # The database is not affected by this exception.
    assert len(comm.inventory_db.get_all_coordinates()) == 2
    comm.inventory_db.remove_coordinate_less_stations()
    assert comm.inventory_db.get_all_coordinates() == {
        "AA.BB": {
            "latitude": 1.0, "longitude": 2.0, "elevation_in_m": 3.0,
            "local_depth_in_m": None}}


def test_zeros_are_valid_coordinates(comm):
    comm.inventory_db.save_station_coordinates(
        station_id="AA.BB", latitude=0.0, longitude=0.0, elevation_in_m=0.0,
        local_depth_in_m=0.0)
    comm.inventory_db.save_station_coordinates(
        station_id="CC.DD", latitude=0.0, longitude=0.0, elevation_in_m=0.0,
        local_depth_in_m=None)
    assert comm.inventory_db.get_all_coordinates() == {
        "AA.BB": {
            "latitude": 0.0, "longitude": 0.0, "elevation_in_m": 0.0,
            "local_depth_in_m": 0.0},
        "CC.DD": {
            "latitude": 0.0, "longitude": 0.0, "elevation_in_m": 0.0,
            "local_depth_in_m": None}}


test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))),
    "data", "inventory_db_download_example.xml")


# Mock object returned by urlopen().
class ns(namedtuple("ns", ["data", "code"])):
    def read(self):
        return self.data


def _urlopen_iris(url):
    if "iris" in url.lower():
        with open(test_file, "rb") as fh:
            data = fh.read()
        return ns(data, 200)
    return ns("", 404)


def _urlopen_orfeus(url):
    if "orfeus" in url.lower():
        with open(test_file, "rb") as fh:
            data = fh.read()
        return ns(data, 200)
    return ns("", 404)


def _urlopen_everything_fails(url):
    return ns("", 404)


def _urlopen_raises(url):
    raise urllib2.HTTPError


@mock.patch("urllib2.urlopen")
def test_coordinate_downloading_iris(patch, comm):
    patch.side_effect = _urlopen_iris
    assert comm.inventory_db.get_coordinates("BW.ROTZ") == \
        {"latitude": 49.766899, "longitude": 12.207,
         "elevation_in_m": 430.0, "local_depth_in_m": None}
    assert comm.inventory_db.get_coordinates("BW.ROTZ") == \
        {"latitude": 49.766899, "longitude": 12.207,
         "elevation_in_m": 430.0, "local_depth_in_m": None}
    # Assert that it has only be called once.
    assert patch.call_count == 1


@mock.patch("urllib2.urlopen")
def test_coordinate_downloading_orfeus(patch, comm):
    patch.side_effect = _urlopen_orfeus
    assert comm.inventory_db.get_coordinates("BW.ROTZ") == \
        {"latitude": 49.766899, "longitude": 12.207,
         "elevation_in_m": 430.0, "local_depth_in_m": None}
    # Will have been called twice. First for IRIS, then for ORFEUS.
    assert patch.call_count == 2
    assert comm.inventory_db.get_coordinates("BW.ROTZ") == \
        {"latitude": 49.766899, "longitude": 12.207,
         "elevation_in_m": 430.0, "local_depth_in_m": None}
    # Not called again.
    assert patch.call_count == 2


@mock.patch("urllib2.urlopen")
def test_coordinate_downloading_fails(patch, comm):
    patch.side_effect = _urlopen_everything_fails
    assert comm.inventory_db.get_coordinates("BW.ROTZ") == \
        {"latitude": None, "longitude": None,
         "elevation_in_m": None, "local_depth_in_m": None}
    assert patch.call_count == 2
    assert comm.inventory_db.get_coordinates("BW.ROTZ") == \
        {"latitude": None, "longitude": None,
         "elevation_in_m": None, "local_depth_in_m": None}
    # Will not have been called again.
    assert patch.call_count == 2

    # If the non-existing station coordiantes are cleared it will be called
    # again.
    comm.inventory_db.remove_coordinate_less_stations()
    assert comm.inventory_db.get_coordinates("BW.ROTZ") == \
        {"latitude": None, "longitude": None,
         "elevation_in_m": None, "local_depth_in_m": None}
    assert patch.call_count == 4


@mock.patch("urllib2.urlopen")
def test_coordinate_downloading_urllib_errors(patch, comm):
    patch.side_effect = _urlopen_raises
    assert comm.inventory_db.get_coordinates("BW.ROTZ") == \
        {"latitude": None, "longitude": None,
         "elevation_in_m": None, "local_depth_in_m": None}
    assert patch.call_count == 2
