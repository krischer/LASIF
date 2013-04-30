#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download and fix the QuakeML files provided by the IRIS SPUD momenttensor
service:

http://www.iris.edu/spud/momenttensor

Downloads the files and saves them as a QuakeML file. You can either
refer to the site displaying the event or the quakeml url.

**Usage:**

Use the SPUD webinterface to search for an event and copy the URL for one
event. Now execute the **iris2quakeml** command with that URL. The file will be
downloaded and stored as a QuakeML 1.2 file in the current folder.

.. code-block:: bash

    $ iris2quakeml http://www.iris.edu/spud/momenttensor/1055532
    Downloading http://www.iris.edu/spudservice/momenttensor/1055532/quakeml...
    Written file GCMT_event_NEAR_EAST_COAST_OF_HONSHU,_JAPAN_Mag_5.5_2013-1-8\
-7-51.xml



Requirements:
    * `ObsPy <http://obspy.org>`_ >= 0.8.3
    * `Requests <http://python-requests.org>`_


Before a file is written, several things are done:

    * The file will be converted to QuakeML 1.2
    * It will only contain one origin (the preferred one)
    * It will only contain one focal mechanism (the preferred one)
    * All units will be converted from dyn*cm to N*m
    * The magnitude will be replace by the moment magnitude calculated from the
      seismic moment specified in the file.

In case anything does not work an error will be raised.


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import argparse
import HTMLParser
import math
from obspy import readEvents
from obspy.core.event import Catalog, Magnitude
from obspy.core.util.geodetics import FlinnEngdahl
import os
import requests
from StringIO import StringIO


def iris2quakeml(url, output_folder=None):
    if not "/spudservice/" in url:
        url = url.replace("/spud/", "/spudservice/")
        if url.endswith("/"):
            url += "quakeml"
        else:
            url += "/quakeml"
    print "Downloading %s..." % url
    r = requests.get(url)
    if r.status_code != 200:
        msg = "Error Downloading file!"
        raise Exception(msg)

    # For some reason the quakeml file is escaped HTML.
    h = HTMLParser.HTMLParser()

    data = h.unescape(r.content)

    # Replace some XML tags.
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

    if ev.preferred_origin():
        ev.origins = [ev.preferred_origin()]
    else:
        ev.origins = [ev.origins[0]]
    if ev.preferred_focal_mechanism():
        ev.focal_mechanisms = [ev.preferred_focal_mechanism()]
    else:
        ev.focal_mechanisms = [ev.focal_mechanisms[0]]

    try:
        mt = ev.focal_mechanisms[0].moment_tensor
    except:
        msg = "No moment tensor found in file."
        raise ValueError
    seismic_moment_in_dyn_cm = mt.scalar_moment
    if not seismic_moment_in_dyn_cm:
        msg = "No scalar moment found in file."
        raise ValueError(msg)

    # Create a new magnitude object with the moment magnitude calculated from
    # the given seismic moment.
    mag = Magnitude()
    mag.magnitude_type = "Mw"
    mag.origin_id = ev.origins[0].resource_id
    # This is the formula given on the GCMT homepage.
    mag.mag = (2.0 / 3.0) * (math.log10(seismic_moment_in_dyn_cm) - 16.1)
    mag.resource_id = ev.origins[0].resource_id.resource_id.replace("Origin",
        "Magnitude")
    ev.magnitudes = [mag]
    ev.preferred_magnitude_id = mag.resource_id

    # Convert the depth to meters.
    org = ev.origins[0]
    org.depth *= 1000.0
    if org.depth_errors.uncertainty:
        org.depth_errors.uncertainty *= 1000.0

    # Ugly asserts -- this is just a simple script.
    assert(len(ev.magnitudes) == 1)
    assert(len(ev.origins) == 1)
    assert(len(ev.focal_mechanisms) == 1)

    # All values given in the QuakeML file are given in dyne * cm. Convert them
    # to N * m.
    for key, value in mt.tensor.iteritems():
        if key.startswith("m_") and len(key) == 4:
            mt.tensor[key] /= 1E7
        if key.endswith("_errors") and hasattr(value, "uncertainty"):
            mt.tensor[key].uncertainty /= 1E7
    mt.scalar_moment /= 1E7
    if mt.scalar_moment_errors.uncertainty:
        mt.scalar_moment_errors.uncertainty /= 1E7
    p_axes = ev.focal_mechanisms[0].principal_axes
    for ax in [p_axes.t_axis, p_axes.p_axis, p_axes.n_axis]:
        if ax is None or not ax.length:
            continue
        ax.length /= 1E7

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
    cat.resource_id = ev.origins[0].resource_id.resource_id.replace("Origin",
        "EventParameters")
    cat.append(ev)
    if output_folder:
        event_name = os.path.join(output_folder, event_name)
    cat.write(event_name, format="quakeml", validate=True)
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
