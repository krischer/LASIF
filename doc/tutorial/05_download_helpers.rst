Download Helpers
----------------

.. note::

    This part is not actually necessary for the tutorial but most likely for
    any real work application.


LASIF comes with a collection of scripts that help downloading waveform and
station data from the IRIS and ArcLink services. Waveform data will always be
downloaded as MiniSEED. Station data will, due to the different products of the
dataservices, either be downloaded as StationXML (IRIS) or dataless SEED.
Furthermore, as many tools so far are not able to deal with StationXML data,
the RESP files for each channel will also be downloaded. This is redundant
information but enables the use of many tools otherwise not possible.

.. caution::

    IRIS changed their web services to the FDSN web service standard. LASIF
    will need to be adopted. Work on this is almost done. In the mean time
    the following commands will only be able to download data from the
    European ArcLink network.

Downloading Data
^^^^^^^^^^^^^^^^

Data are downloaded on a per event basis. The **config.xml** file contains
some specification to detail the download.

To download the data for an event, choose one and run

.. code-block:: bash

    $ lasif download_data GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11


The command essentially just tries to download everything it can, both the 
waveforms and station metadata. It queries
the IRIS DMC and ArcLink for all stations available in the physical domain and
then downloads the appropriate data. It accounts for the domain borders and
possible domain rotations. It is influenced by three parameters in the
**config.xml** file:

* The *arclink_username* tag should be your email. It will be send with all
  requests to the ArcLink network. They ask for it in case they have to contact
  you for whatever reason. Please provide a real email address. Must not be
  empty.
* *seconds_before_event*: Used by the waveform download scripts. It will
  attempt to download this many seconds for every waveform before the origin of
  the associated event.
* *seconds_after_event*: Used by the waveform download scripts. It will attempt
  to download this many seconds for every waveform after the origin of the
  associated event. Adapt this to the size of your inversion domain.

Depending on the domain size, event location, and origin time, this can take a
while. Executing the same command again will only attempt to download data not
already present. All data will be placed in `DATA/EVENT_NAME/raw`.

.. note::

    At this point it is worth mentioning that LASIF keeps logs of many actions
    that the user performs. All logs will be saved in the *LOGS* subfolder.


.. Downloading Station Metadata
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. LASIF also includes some functionality to download station metadata. It will,
.. download RESP files from IRIS and dataless SEED files from ArcLink. It works
.. the same as it does for the waveforms. To download all stations for one event
.. simply execute

.. .. code-block:: bash

..    $ lasif download_stations GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11


.. The `lasif download_stations` command will, for the specified event, figure
.. what waveform data is present in the `DATA/EVENT_NAME/raw` folder and download
.. all missing station metadata information for these files.

.. note::

    At some point in the near future the station metadata downloading routines
    will be changed so that they exclusively work with StationXML metadata.


.. note::

    The automatic download routines included in LASIF are sometimes not able to
    find stations files. IRIS as well as Orfeus have FTP servers with anonymous
    access that potentially provide missing station files:

    * ftp://www.orfeus-eu.org/pub/data/metadata
    * http://www.iris.edu/pub/RESPONSES/

    Just make sure to adhere to the naming scheme imposed by LASIF.

