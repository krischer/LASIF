.. centered:: Last updated on *February 22nd 2018*.

Data Inspection
---------------

Once waveform and station metadata has been downloaded (either with the
built-in helpers or manually) and placed in the correct folders, **LASIF** can
start to work with it. To continue this tutorial, you can copy your asdf files
from the **LASIF** folder. You can also download data automatically for the
events and use that data to continue the tutorial. This is how you could copy
the data from the **LASIF** folder if you are positioned in the projects root
folder:

.. code-block:: bash

    $ cp {lasif_code_folder}/lasif/tests/data/ExampleProject/DATA/EARTHQUAKES/* ./DATA/EARTHQUAKES/


.. note::

    **LASIF** essentially needs three ingredients to be able to interpret waveform
    data:

    * The actual waveforms
    * The location of the recording seismometer
    * The instrument response for each channel at the time of data recording

    This information needs to be existing in the ASDF file for each event.
    We will make a tutorial showing the way the data needs to be structured
    later.


At this point, **LASIF** is able to match available station and waveform
information. Only stations where the three aforementioned ingredients are
available will be considered to be stations that are good to be worked with by
**LASIF**. Others will be ignored.

Once you have picked a few events it is possible to plot them all on a single
overview plot by executing:

.. code-block:: bash

    $ lasif plot_events

To get an overview, of what data is actually available for any given event,
just execute:

.. code-block:: bash

    $ lasif event_info -v GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15

    Earthquake with 5.9 Mwc at TURKEY
        Latitude: 39.150, Longitude: 29.100, Depth: 7.0 km
        2011-05-19T20:15:22.900000Z UTC

    Station and waveform information available at 4 stations:

    ============================================================
                 ID       Latitude      Longitude Elevation_in_m
    ============================================================
            GE.TIRR      44.458099        28.4128           77.0
             II.KIV      43.955299      42.686298         1054.0
             MN.IDI      35.287998      24.889999          750.0
            YD.4F14      44.988819      24.610781          286.0

.. note::

    The local depth is allowed to not be set. In that case it will be
    assumed to be zero. For all practical purposes the local depth does not
    matter for continental scale inversions.


It is furthermore possible to plot the availability information for one event
including a very simple ray coverage plot with:

.. code-block:: bash

    $ lasif plot_event GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15


Now stations have an even weight and would each have the same effect when
computing adjoint sources. A method to change that would be to weigh the
stations up or down depending on the average distance to neighbouring stations.

.. code-block:: bash

    $ lasif compute_station_weights A

This can also be done for one event at a time. It writes the weighting factors
to a toml file in **/SETS/WEIGHTS/WEIGHTS_A/WEIGHTS_A.toml** where A is the
name of the weight set specified when command was run. This file can be used to
manually modify station weights. Now we can plot the event again but showing on
a colour scale how the stations are weighted.

.. code-block:: bash

    $ lasif plot_event GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15 --weight_set_name A

Now we can see how the stations are colour coded according to the weighting
scheme. This method is more relevant when you have more stations where their
distribution is far from uniform.

If you are interested in getting a coverage plot of all events and data
available for the current project, please execute the ``plot_raydensity``
command:

.. code-block:: bash

    $ lasif plot_raydensity

Actually plotting this may take a fair while, depending on the amount of data
you have.
Keep in mind that this only results in a reasonable plot for large amounts of
data; for the toy example used in the tutorial it will not work. It is not a
physically accurate plot but helps in judging data coverage and directionality
effects. An example from a larger **LASIF** project illustrates this:


.. image:: ../images/raydensity.jpg
    :width: 70%
    :align: center



