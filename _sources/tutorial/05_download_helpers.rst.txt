.. centered:: Last updated on *August 12th 2016*.

.. note::

    The following links shows the example project as it should be just before
    step 4. You can use this to check your progress or restart the tutorial at
    this very point.

    `After Step 5: Waveform Data <https://github.com/krischer/LASIF_Tutorial/tree/after_step_5_waveform_data>`_

Download Helpers
----------------

.. note::

    This part is not actually necessary for the tutorial but most likely for
    any real work application.


**LASIF** comes with a collection of scripts that help downloading waveform and
station data from all data centers implementing the FDSN web services. This
largely amounts to data from Europe, Northern America, Brazil, and New Zealand.
If your inversion requires data from other places, you'll have to retrieve it
on your own.

Downloading Data
^^^^^^^^^^^^^^^^

Data are downloaded on a per event basis. The ``config.xml`` file contains
some specification to detail the download.

To download the data for an event, choose one and run

.. code-block:: bash

    $ lasif download_data GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11


The command just tries to download everything it can within your chosen domain,
both the waveforms and station metadata. It queries successively queries all
known FDSN data centers and integrates all data. It accounts for the domain
borders and possible domain rotations. It is furthermore influenced by the
following parameters in the ``config.xml`` file:

* ``seconds_before_event``: Used by the waveform download scripts. It will
  attempt to download this many seconds for every waveform before the origin of
  the associated event.
* ``seconds_after_event``: Used by the waveform download scripts. It will
  attempt to download this many seconds for every waveform after the origin of
  the associated event. Adapt this to the size of your inversion domain.
* ``interstation_distance_in_m``: The minimum distance between two stations so
  that data from both would be downloaded in meters. If stations are closer
  than that only one is downloaded.
* ``channel_priorities``: The priority in which channels will be chosen from
  each station.
* ``location_priorities``: The priority in which location codes will be chosen
  from each station.

Depending on the domain size, event location, and origin time, this can take a
while. Executing the same command again will only attempt to download data not
already present. All waveform data will be placed in ``DATA/EVENT_NAME/raw``
and all station data in ``STATIONS/StationXML``.

.. note::

    At this point it is worth mentioning that **LASIF** keeps logs of many
    actions that the user performs. All logs will be saved in the ``LOGS``
    subfolder.
