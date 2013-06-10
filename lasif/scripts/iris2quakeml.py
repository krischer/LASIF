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
    * It will only contain one focal mechanism (the preferred one)
    * All units will be converted from dyn*cm to N*m

In case anything does not work an error will be raised.


This is rather unstable due to potential changes in the SPUD webservice.


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import argparse
import HTMLParser
from obspy import readEvents
from obspy.core.util.geodetics import FlinnEngdahl
import os
import re
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

    # Replace the duplice moment tensor publicID with a proper one.
    data = re.sub(r"(<momentTensor\s*publicID=\".*)focalmechanism(.*\">)",
        r"\1momenttensor\2", data)

    data = StringIO(data)

    try:
        cat = readEvents(data)
    except:
        msg = "Could not read downloaded event data"
        raise ValueError(msg)

    cat.events = cat.events[:1]
    ev = cat[0]

    # Parse the event and get the preferred focal mechanism. Then get the
    # origin and magnitude associated with that focal mechanism. All other
    # focal mechanisms, origins and magnitudes will be removed. Just makes it
    # simpler and less error prone.
    if ev.preferred_focal_mechanism():
        ev.focal_mechanisms = [ev.preferred_focal_mechanism()]
    else:
        ev.focal_mechanisms = [ev.focal_mechanisms[:1]]

    # Some shortcuts.
    foc_mec = ev.focal_mechanisms[0]
    mt = foc_mec.moment_tensor
    tensor = mt.tensor

    # Set the origin and magnitudes of the event.
    ev.magnitudes = [mt.moment_magnitude_id.getReferredObject()]
    ev.origins = [mt.derived_origin_id.getReferredObject()]

    # All values given in the QuakeML file are given in dyne * cm. Convert them
    # to N * m.
    for key, value in tensor.iteritems():
        if key.startswith("m_") and len(key) == 4:
            tensor[key] /= 1E7
        if key.endswith("_errors") and hasattr(value, "uncertainty"):
            tensor[key].uncertainty /= 1E7
    mt.scalar_moment /= 1E7
    if mt.scalar_moment_errors.uncertainty:
        mt.scalar_moment_errors.uncertainty /= 1E7
    p_axes = ev.focal_mechanisms[0].principal_axes
    for ax in [p_axes.t_axis, p_axes.p_axis, p_axes.n_axis]:
        if ax is None or not ax.length:
            continue
        ax.length /= 1E7

    # Check if it has a source time function
    stf = mt.source_time_function
    if stf:
        if stf.type != "triangle":
            msg = ("Source time function type '%s' not yet mapped. Please "
                "contact the developers.") % stf.type
            raise NotImplementedError(msg)
        if not stf.duration:
            if not stf.decay_time:
                msg = "Not known how to derive duration without decay time."
                raise NotImplementedError(msg)
            # Approximate the duraction for triangular STF.
            stf.duration = 2 * stf.decay_time

    # Get the flinn_engdahl region for a nice name.
    fe = FlinnEngdahl()
    region_name = fe.get_region(ev.origins[0].longitude,
        ev.origins[0].latitude)
    region_name = region_name.replace(" ", "_")
    event_name = "GCMT_event_%s_Mag_%.1f_%s-%s-%s-%s-%s.xml" % \
        (region_name, ev.magnitudes[0].mag, ev.origins[0].time.year,
        ev.origins[0].time.month, ev.origins[0].time.day,
        ev.origins[0].time.hour, ev.origins[0].time.minute)

    cat.resource_id = ev.origins[0].resource_id.resource_id.replace("origin",
        "event_parameters")
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
