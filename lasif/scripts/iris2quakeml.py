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


:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2013
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
import argparse
import HTMLParser
from obspy import read_events
import os
from StringIO import StringIO
import urllib2

from lasif.utils import get_event_filename


def iris2quakeml(url, output_folder=None):
    if "/spudservice/" not in url:
        url = url.replace("/spud/", "/spudservice/")
        if url.endswith("/"):
            url += "quakeml"
        else:
            url += "/quakeml"
    print "Downloading %s..." % url

    r = urllib2.urlopen(url)
    if r.code != 200:
        r.close()
        msg = "Error Downloading file!"
        raise Exception(msg)

    # For some reason the quakeml file is escaped HTML.
    h = HTMLParser.HTMLParser()

    data = h.unescape(r.read())
    r.close()

    data = StringIO(data)

    try:
        cat = read_events(data)
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

    # Set the origin and magnitudes of the event.
    mt = ev.focal_mechanisms[0].moment_tensor
    ev.magnitudes = [mt.moment_magnitude_id.get_referred_object()]
    ev.origins = [mt.derived_origin_id.get_referred_object()]

    event_name = get_event_filename(ev, "GCMT")

    if output_folder:
        event_name = os.path.join(output_folder, event_name)
    cat.write(event_name, format="quakeml", validate=True)
    print "Written file", event_name


def main():
    parser = argparse.ArgumentParser(description=(
        "Download the QuakeML files provided by the IRIS SPUD "
        "momenttensor service. Will be saved as a QuakeML file in the "
        "current folder."))
    parser.add_argument("url", metavar="U", type=str,
                        help="The URL to download.")
    args = parser.parse_args()

    url = args.url
    iris2quakeml(url)


if __name__ == "__main__":
    main()
