#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download and fix the QuakeML files provided by the IRIS SPUD momenttensor
service

http://www.iris.edu/spud/momenttensor

Downloads the files and saves them as a QuakeML file. You can either
refer to the site displaying the event or the quakeml url.

=============================================
=============================================
Usage:

python iris2quakeml.py http://www.iris.edu/spud/momenttensor/1055532

or

python iris2quakeml.py \
    http://www.iris.edu/spudservice/momenttensor/1055532/quakeml
=============================================
=============================================

Requirements:
    * ObsPy >= 0.8.3
    * Requests

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""
import argparse
import HTMLParser
import requests
from obspy import readEvents
from obspy.core.event import Catalog
from obspy.core.util.geodetics import FlinnEngdahl
from StringIO import StringIO


def iris2quakeml(url):
    if not "/spudservice/" in url:
        url = url.replace("/spud/", "/spudservice/")
        if url.endswith("/"):
            url += "quakeml"
        else:
            url += "/quakeml"
    print "Downloading %s..." % url
    r = requests.get(url)
    if r.status_code != 200:
        msg = "Error Downloading file"
        raise Exception(msg)

    # For some reason the quakeml file is escaped HTML.
    h = HTMLParser.HTMLParser()

    data = h.unescape(r.content)
    data = data.replace("long-period body waves", "body waves")
    data = data.replace("intermediate-period surface waves", "surface waves")
    data = data.replace("long-period mantle waves", "mantle waves")

    data = data.replace("<html><body><pre>", "")
    data = data.replace("</pre></body></html>", "")

    data = StringIO(data)

    try:
        cat = readEvents(data)
    except:
        msg = "Could not read downloaded event data"
        raise ValueError(msg)

    # Parse the event, and use only one origin, magnitude and focal mechanism.
    # Only the first event is used. Should not be a problem for the chosen
    # global cmt application.
    ev = cat[0]

    if ev.preferred_magnitude():
        ev.magnitudes = [ev.preferred_magnitude()]
    else:
        ev.magnitudes = [ev.magnitudes[0]]

    if ev.preferred_origin():
        ev.origins = [ev.preferred_origin()]
    else:
        ev.origins = [ev.origins[0]]
    if ev.preferred_focal_mechanism():
        ev.focalMechanisms = [ev.preferred_focal_mechanism()]
    else:
        ev.focalMechanisms = [ev.focalMechanisms[0]]

    # Ugly asserts -- this is just a simple script.
    assert(len(ev.magnitudes) == 1)
    assert(len(ev.origins) == 1)
    assert(len(ev.focal_mechanisms) == 1)

    mt = ev.focal_mechanisms[0].moment_tensor.tensor

    for key, value in mt.iteritems():
        if key.startswith("m_") and len(key) == 4:
            mt[key] /= 1E7
        if key.endswith("_errors") and hasattr(value, "uncertainty"):
            mt[key].uncertainty /= 1E7

    # Get the flinn_engdahl region for a nice name.
    fe = FlinnEngdahl()
    region_name = fe.get_region(ev.origins[0].longitude,
        ev.origins[0].latitude)
    region_name = region_name.replace(" ", "_")
    event_name = "GCMT_event_%s_Mag_%.1f_%s-%s-%s-%s-%s.xml" % \
        (region_name, ev.magnitudes[0].mag, ev.origins[0].time.year,
        ev.origins[0].time.month, ev.origins[0].time.day,
        ev.origins[0].time.hour, ev.origins[0].time.minute)
    cat = Catalog()
    cat.append(ev)
    cat.write(event_name, format="quakeml")
    print "Written file", event_name


def main():
    parser = argparse.ArgumentParser(description=(
        "Download and fix the QuakeML files provided by the IRIS SPUD "
        "momenttensor service. Will be saved as a QuakeML file in the "
        "current folder"))
    parser.add_argument("url", metavar="U", type=str,
                       help="The URL to download.")
    args = parser.parse_args()

    url = args.url
    iris2quakeml(url)

if __name__ == "__main__":
    main()
