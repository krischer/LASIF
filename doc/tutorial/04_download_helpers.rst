.. centered:: Last updated on *February 22nd 2018*.

Download Helpers
----------------

.. note::

    This part is not actually necessary for the tutorial but most likely for
    any real work application.

**LASIF** comes with a collection of scripts that help downloading waveform and
station data from all data centers implementing the FDSN web services. This
data is not evenly distributed around the globe and largely amounts to data
from Europe, Northern America and New Zealand. There is data from other areas
but they are usually more sparsely distributed. You are free to add your own
data into the event files using the pyasdf package. We will link to a tutorial
on how to do that here once we make one. Hopefully soon. **LASIF** does have
some scripts that attempt to reduce the effect of uneven data distribution
which will be explained later.

Downloading Data
^^^^^^^^^^^^^^^^

Data are downloaded on a per event basis. The *lasif_config.toml* file
contains some specification to detail the download.

To download the data for an event, choose one and run

.. code-block:: bash

    $ lasif download_data GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15


The command just tries to download everything it can within your chosen domain,
both the waveforms and station metadata. It queries all known FDSN data centers
and integrates all data. It accounts for the domain borders to see whether
stations fit inside the domain. It is furthermore influenced by the
following parameters in the *lasif_config.toml* file:

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
already present. All waveform and station data will be placed in
*DATA/EARTHQUAKES/{event_name}*.

The ``download_data`` command has an option to download only from a specific
provider like IRIS for example.

When using many events and a large domain, keeping all the raw data can be
problematic if you are working on a local computer. **LASIF** currently has
a method to downsample the downloaded data and keep it as raw data. We do
not recommend using this option since it will introduce small differences
in your waveforms due to filtering. If you want to use this option you
have to keep in mind that you are loosing some of the accuracy of your
method which reduces the reliability of your inversion. Support of this
method might be removed in the future as it was an experiment.

.. note::

    At this point it is worth mentioning that **LASIF** keeps logs of many
    actions that the user performs. All logs will be saved in the
    ``OUTPUT/LOGS`` subfolder.
