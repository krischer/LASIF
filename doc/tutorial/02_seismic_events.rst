Seismic Events
--------------
Once the domain has been adjusted to your needs, you need to tell **LASIF**
which events you want to use for the inversion. This works by simply placing a
valid QuakeML 1.2 file at the correct location.

All events have to be stored in the ``EVENTS`` subfolder of the project. They
have to be QuakeML 1.2 files with full moment tensor information.

LASIF provides some convenience methods for this purpose. One can leverage the
`IRIS SPUD service <http://www.iris.edu/spud/momenttensor>`_ to get GlobalCMT
events.  Simply search for an event on their webpage and copy the event url.
The ``lasif add_spud_event`` command will then grab the QuakeML file from the
url and store an XML file in the correct folder.


.. code-block:: bash

    $ lasif add_spud_event http://www.iris.edu/spud/momenttensor/835040
    $ lasif add_spud_event http://www.iris.edu/spud/momenttensor/912955


These commands will download two event files and store them in the
``EVENTS`` subfolder.

.. code-block:: bash

    $ ls EVENTS

    GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17.xml
    GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20.xml

A more convenient way to see which events are currently defined in the
project is the ``lasif list_events`` command.

.. code-block:: bash

    $ lasif list_events

    2 events in project:
    +------------------------------------------------------------+--------------------------+---------------------+
    | Event Name                                                 |  Lat/Lng/Depth(km)/Mag   | # raw/preproc/synth |
    +------------------------------------------------------------+--------------------------+---------------------+
    | GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17             |   44.9 / 8.5  / 10 / 4.9 |    0 /     0 /    0 |
    | GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20 |   43.3 / 20.8 /  9 / 5.9 |    0 /     0 /    0 |
    +------------------------------------------------------------+------------------------+-----------------------+

You will notice that events are identified via their filename minus the
extension. This is an easy and flexible solution enabling you to tag the events
as you see fit. The slight disadvantage of this approach is that **you must not
change the event filenames after you have worked with them** because all
additional information for that event will be related to it via the event
filename. So please give them a good and reasonable filename. If you really
feel that event renaming is a necessary feature please file an issue on Github
so that the authors can add a proper event renaming function.

The ``lasif plot_events`` command will show a map with all events currently
part of the project.

.. code-block:: bash

    $ lasif plot_events

.. plot::

    import lasif.visualization
    from obspy import UTCDateTime
    bmap = lasif.visualization.plot_domain(-10.0, 10.0, -10.0, 10.0, 2.5,
        rotation_axis=[1.0, 1.0, 0.2], rotation_angle_in_degree=-65.0,
        plot_simulation_domain=False, zoom=True)
    events = [{
        'depth_in_km': 9.0,
        'event_name': 'GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20-2',
        'filename': 'GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20-2.xml',
        'latitude': 43.29,
        'longitude': 20.84,
        'm_pp': -4.449e+17,
        'm_rp': -5.705e+17,
        'm_rr': -1.864e+17,
        'm_rt': 2.52e+16,
        'm_tp': 4.049e+17,
        'm_tt': 6.313e+17,
        'magnitude': 5.9,
        'magnitude_type': 'Mwc',
        'origin_time': UTCDateTime(1980, 5, 18, 20, 2, 57, 500000),
        'region': u'NORTHWESTERN BALKAN REGION'
    }, {
        'depth_in_km': 10.0,
        'event_name': 'GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17-14',
        'filename': 'GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17-14.xml',
        'latitude': 44.87,
        'longitude': 8.48,
        'm_pp': 1.189e+16,
        'm_rp': -1600000000000000.0,
        'm_rr': -2.271e+16,
        'm_rt': -100000000000000.0,
        'm_tp': -2.075e+16,
        'm_tt': 1.082e+16,
        'magnitude': 4.9,
        'magnitude_type': 'Mwc',
        'origin_time': UTCDateTime(2000, 8, 21, 17, 14, 27),
        'region': u'NORTHERN ITALY'}]
    lasif.visualization.plot_events(events, bmap)


The ``lasif event_info`` command is your friend if you desire more information
about a certain event:

.. code-block:: bash

    $ lasif event_info GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17-14


    Earthquake with 4.9 Mwc at NORTHERN ITALY
    Latitude: 44.870, Longitude: 8.480, Depth: 10.0 km
    2000-08-21T17:14:27.000000Z UTC

    Station and waveform information available at 51 stations. Use '-v' to print them.



The information given with this command will be the one LASIF uses. This is
useful if the event has more then one origin and you want to know which one
LASIF actually uses. Notice that the event currently has no data associated
with it. We will fix this in the next section.

.. note::

    You do not need to add all events you plan to use in the inversion at the
    beginning. Only add those you want to use for the very first inversion.
    LASIF is rather flexible and enables you to use different events, data,
    weighting schemes, ... for every iteration. It will keep track of what
    actually happened during each iteration so the project gains
    **reproducibility and provenance**.


