#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper function to bin great-circle paths.

Useful for ray-density plots.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from collections import namedtuple
from geographiclib import geodesic
import numpy as np


Point = namedtuple("Point", ["lat", "lng"])


class Range(namedtuple("Range", ["min", "max", "count"])):
    @property
    def delta(self):
        return self.range / (self.count - 1)

    @property
    def range(self):
        return float(self.max - self.min)


class GreatCircleBinner(object):
    def __init__(self, min_lat, max_lat, lat_count, min_lng, max_lng,
            lng_count):
        self.lats = Range(min_lat, max_lat, lat_count)
        self.lngs = Range(min_lng, max_lng, lng_count)
        self.max_range = max(self.lats.range, self.lngs.range)
        self.bins = np.zeros((self.lngs.count, self.lats.count),
            dtype="uint32")

    def add_point(self, point):
        """
        Adds a single point an increments the value at that point by one.
        """
        # Skip points outside of the range.
        if not (self.lngs.min <= point.lng <= self.lngs.max) or \
                not (self.lats.min <= point.lat <= self.lats.max):
            return

        lng_index = int(round(((point.lng - self.lngs.min) / self.lngs.range) *
            (self.lngs.count - 1)))
        lat_index = int(round(((point.lat - self.lats.min) / self.lats.range) *
            (self.lats.count - 1)))
        self.bins[lng_index, lat_index] += 1

    def add_greatcircle(self, point_1, point_2, max_npts=3000):
        point = geodesic.Geodesic.WGS84.Inverse(lat1=point_1.lat,
            lon1=point_1.lng, lat2=point_2.lat, lon2=point_2.lng)
        line = geodesic.Geodesic.WGS84.Line(
            point_1.lat, point_1.lng, point["azi1"])

        npts = int((point["a12"] / self.max_range) * max_npts)
        for i in xrange(npts + 1):
            line_point = line.Position(i * point["s12"] / float(npts))
            self.add_point(Point(line_point["lat2"], line_point["lon2"]))

    @property
    def coordinates(self):
        return np.meshgrid(
            np.linspace(self.lngs.min, self.lngs.max, self.lngs.count),
            np.linspace(self.lats.min, self.lats.max, self.lats.count))
