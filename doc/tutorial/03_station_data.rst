.. centered:: Last updated on *August 12th 2016*.

.. note::

    The following links shows the example project as it should be just before
    step 3. You can use this to check your progress or restart the tutorial at
    this very point.

    `After Step 3: Seismic Events <https://github.com/krischer/LASIF_Tutorial/tree/after_step_3_seismic_events>`_

Station Data
------------

**LASIF** needs to know the coordinates and instrument response of each channel.
One way to achieve this, is to use SAC files, which contain coordinates, and RESP
files containing the response information for each channel. Another possibility
is to use MiniSEED waveform data and the corresponding dataless SEED or
StationXML files. Please keep in mind that LASIF currently expects to only have
channels of one station in each dataless SEED and StationXML file. All station
information files are stored in a subfolder of the ``STATIONS`` directory.

For more information about how LASIF derives station coordinates:
:doc:`../how_lasif_finds_coordinates`

.. caution::

    If possible in any way, we strongly recommend to use station data in the
    ``StationXML`` format. It is the most future proof and sane format of
    the three. Please only use ``RESP`` files if absolutely necessary. The
    problem with them is that they do not contain coordinates and thus LASIF
    needs to derive them by other means which is not as clean and more error
    prone.


Naming scheme
^^^^^^^^^^^^^

**dataless SEED**

All dataless SEED files are expected to be in the ``STATIONS/SEED`` directory
and be named after the following scheme::

    dataless.NETWORK_STATION[.X]

``NETWORK`` and ``STATION`` should be replaced with the corresponding network
and stations codes. It is possible that multiple files are needed for each
station (e.g. different files for different time intervals/channels) and thus
``.1``, ``.2``, ... can be appended to the filename. LASIF will automatically
choose the correct file in case they need to be accessed.

**StationXML**

All StationXML files are expected to be placed in the ``STATIONS/StationXML``
folder and follow the scheme::

    station.NETWORK_STATION[.X].xml

The logic for for the different parts is the same as for the dataless SEED
files described in the previous paragraph.

**RESP Files**

All RESP files are to be put in the ``STATIONS/RESP`` folder with the
following name::

    RESP.NETWORK.STATION.LOCATION.CHANNEL[.X]

In contrast to the two other station information formats the RESP filename also
has to include the location and channel identifiers.


Stations for Tutorial
^^^^^^^^^^^^^^^^^^^^^

For this tutorial we will work with a bunch of well distributed stations in
the domain of interest. Unpack
:download:`this archive <../downloads/stations.tar.bz2>` and place the XML
files in the ``STATIONS/StationXML/`` folder. At this point, LASIF is aware
of the stations. Please note that the stations will not yet show up in any
command as **LASIF only treats data that has raw waveforms and an associated
station file as being actually available**.
