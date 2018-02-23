.. centered:: Last updated on *February 22nd 2018*.

Seismic Events
--------------
Once you have made a mesh file and referenced its location in the config file
it is time to import events to the project. **LASIF** expects the data to be
stored in the **DATA/EARTHQUAKES/** directory in an `ASDF Data format
<http://seismicdata.github.io/pyasdf/>`_. You can either use the pyasdf package
to organise your own data into the ASDF format or you can use the tools that
**LASIF** provides to query different databases for events within your domain.
**LASIF** expects the data to contain valid QuakeML files with full moment
tensor information. Each event is organised in one asdf file. The waveform and
station data for said event will also be included in that one file. That means
that all the raw information concerning this one event is contained in one
file. That makes bookkeeping easier and the pyasdf can be used to look into
each file.

LASIF provides some convenience methods for this purpose. One can make use of the
`IRIS SPUD service <http://www.iris.edu/spud/momenttensor>`_ to get GlobalCMT
events.  Simply search for an event on their webpage and copy the event url.
The ``lasif add_spud_event`` command will then grab the QuakeML file from the
url and store an asdf file in the correct folder.


.. code-block:: bash

    $ lasif add_spud_event http://ds.iris.edu/spud/momenttensor/988455
    $ lasif add_spud_event http://ds.iris.edu/spud/momenttensor/735711


These commands will download two event files and store them in the
``EVENTS`` subfolder.

.. code-block:: bash

    $ ls DATA/EARTHQUAKES

    GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11.h5
    GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-11.h5

A more convenient way to see which events are currently defined in the
project is the ``lasif list_events`` command.

.. code-block:: bash

    $ lasif list_events

    2 events in project:
    +-------------------------------------------+-----------------------------+
    | Event Name                                |    Lat/Lng/Depth(km)/Mag    |
    +-------------------------------------------+-----------------------------+
    | GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11 |   38.8 /   40.1 /   4 / 5.1 |
    | GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15 |   39.1 /   29.1 /   7 / 5.9 |
    +----------------------------------------+--------------------------------+

You will notice that events are identified via their filename minus the
extension. This is an easy and flexible solution enabling you to tag the events
as you see fit. The slight disadvantage of this approach is that **you must not
change the event filenames after you have worked with them** because all
additional information for that event will be related to it via the event
filename. So please give them a good and reasonable filename. If you really
feel that event renaming is a necessary feature please file an issue on Github
so that the authors can add a proper event renaming function.

The ``lasif plot_events`` command will show a map with all events currently
part of the project. With the same command, you can get histograms of depth
distribution and origin time distribution by appending ``--type depth`` or
``--type time``, respectively.

.. code-block:: bash

    $ lasif plot_events


The ``lasif event_info`` command is your friend if you want more information
about a certain event:

.. code-block:: bash

    $ lasif event_info GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11

    Earthquake with 5.1 Mwc at TURKEY
        Latitude: 38.690, Longitude: 39.990, Depth: 28.9 km
        2010-03-24T14:11:34.300000Z UTC

    Station and waveform information available at 0 stations. Use '-v' to print them.

The information given with this command will be what **LASIF** uses. This is
useful if the event has more than one origin and you want to know which one
is actually used by **LASIF**. Notice that the event currently has no data associated
with it. We will fix this in the next section.

After adding the events, this is how your directory structure should look:

.. code-block:: none

    Tutorial
    ├── ADJOINT_SOURCES
    ├── DATA
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    │       ├── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11.h5
    │       └── GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15.h5
    ├── FUNCTIONS
    │   ├── __init__.py
    │   ├── light_preprocessing.py
    │   ├── preprocessing_function_asdf.py
    │   ├── process_data.py
    │   ├── process_synthetics.py
    │   ├── source_time_function.py
    │   └── window_picking_function.py
    ├── GRADIENTS
    ├── MODELS
    ├── OUTPUT
    │   └── LOGS
    ├── PROCESSED_DATA
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    ├── SALVUS_INPUT_FILES
    ├── SETS
    │   ├── WEIGHTS
    │   └── WINDOWS
    ├── SYNTHETICS
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    └── lasif_config.toml


Automatic Event Selection
^^^^^^^^^^^^^^^^^^^^^^^^^

Selecting events becomes tedious when selecting a larger number of events. Thus
**LASIF** comes with an automatic routine to select events from the GCMT
catalog, the ``lasif add_gcmt_events`` command. Arguments are number of events
to select, minimum magnitude, maximum magnitude, and the minimum distance
between two events in kilometers. See its help method for more details.

It will select events, that fit the selected criteria and fit inside the
domain defined by the mesh file, in an optimally distributed fashion by
successively adding events that have the largest distance to the next closest
station, approximating a Poisson disc distribution.

.. code-block:: bash

    $ lasif add_gcmt_events 40 5 6.5 10

    LASIF currently contains GCMT data from 2005 to 2017.
    ...
    Selected 40 events.
    Written EVENTS/GCMT_event_...
    ...

These events will not be used in this tutorial but rather the events already
downloaded using the add_spud_event method. So if you ran the command above
but want to follow the tutorial step by step. We would recommend either
removing the 40 new events or removing all of them and running the
add_spud_event command again like is done above. You can remove all your
events by running

.. code-block:: bash

    $ rm DATA/EARTHQUAKES/*


.. note::

    You do not need to add all events you plan to use in the inversion at the
    beginning. Only add those you want to use for the very first inversion.
    **LASIF** is rather flexible and enables you to use different events, data,
    weighting schemes, etc. for every iteration. It will keep track of what
    actually happened during each iteration so the project gains
    **reproducibility and provenance**.


