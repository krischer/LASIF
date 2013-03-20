.. FWIW documentation master file, created by
   sphinx-quickstart on Fri Feb  1 15:47:43 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FWIW's Documentation!
===================================

FWIW (**F**\ ull **W**\ aveform **I**\ nversion **W**\ orkflow or **F**\ or
**W**\ hat **I**\ 'ts **W**\ orth) is a **data-driven workflow tool** to
perform full waveform inversions.
It is opinionated and strict meaning that it forces a certain data and
directory structure. The upside is that it only requires a very minimal amount
of configuration and maintenance. It attempts to gather all necessary
information from the data itself so there is no need to keep index or content
files.

All parts of FWIW can work completely on their own. See the class and
function documentation at the end of this document. Furthermore FWIW offers
a project based inversion workflow management system which is introduced in the
following tutorial.


Tutorial
========
FWIW works with the notion of so called inversion projects. A project is
defined as a series of iterations working on the same physical domain. Where
possible and useful, FWIW will use XML files to store information. The
reasoning behind this is twofold. It is easily machine and human readable. It
also serves as a preparatory step towards a fully database driven full waveform
inversion workflow as all necessary information is already stored in an easily
indexable data format.

Creating a New Project
----------------------
The necessary first step, whether for starting a new inversion or migrating an
already existing inversion to FWIW, is to create a new project. In the
following the project will be called **MyInversion**.

.. code-block:: bash

    $ fwiw init_project MyInversion

This will create the following directory structure::

    MyInversion
    |-- config.xml
    |-- DATA
    |-- EVENTS
    |-- LOGS
    |-- MODELS
    |-- STATIONS
    |   |-- RESP
    |   |-- SEED
    |   |-- StationXML
    |-- SYNTHETICS


The configuration for each project works , is defined in the **config.xml**
file. It is a simple, self-explanatory XML format. The nature of SES3D's
coordinate system has the effect that simulation is most efficient in
equatorial regions. Thus it is oftentimes advantageous to rotate the frame of
reference so that the simulation happens close to the equator. FWIW first
defines the simulation domain; the actual simulation happens here. Optional
rotation parameters define the physical location of the domain. The coordinate
system for the rotation parameters is described in :py:mod:`fwiw.rotations`.
You will have to edit the file to adjust it to your region of interest. It will
look something like the following.

.. code-block:: xml

    <?xml version="1.0" encoding="utf-8"?>
    <fwiw_project>
        <name>MyInversion</name>
        <description></description>
        <download_settings>
            <arclink_username></arclink_username>
            <seconds_before_event>300</seconds_before_event>
            <seconds_after_event>3600</seconds_after_event>
        </download_settings>
        <domain>
          <domain_bounds>
            <minimum_longitude>-20.0</minimum_longitude>
            <maximum_longitude>20.0</maximum_longitude>
            <minimum_latitude>-20.0</minimum_latitude>
            <maximum_latitude>20.0</maximum_latitude>
            <minimum_depth_in_km>0.0</minimum_depth_in_km>
            <maximum_depth_in_km>200.0</maximum_depth_in_km>
            <boundary_width_in_degree>3.0</boundary_width_in_degree>
          </domain_bounds>
          <domain_rotation>
            <rotation_axis_x>1.0</rotation_axis_x>
            <rotation_axis_y>1.0</rotation_axis_y>
            <rotation_axis_z>1.0</rotation_axis_z>
            <rotation_angle_in_degree>-45.0</rotation_angle_in_degree>
          </domain_rotation>
        </domain>
    </fwiw_project>


It should be fairly self-explanatory.

* The *boundary_width_in_degree* tag is just used for the download helpers. No
  data will be downloaded within *boundary_width_in_degree* distance to the
  domain border. This is useful for e.g. absorbing boundary conditions.
* The *arclink_username* tag should be your email. It will be send with all
  requests to the ArcLink network. They ask for it in case they have to contact
  you for whatever reason. Please provide a real email address. Must not be
  empty.


.. note::

    All **fwiw** commands work and use the correct project as long as they
    are executed somewhere inside a projects folder structure.

At any point you can have a look at the defined domain with

.. code-block:: bash

    $ cd MyInversion
    $ fwiw plot_domain

This will open a window showing the location of the physical domain and the
simulation domain. The inner contours show the domain minus the previously
defined boundary width.

.. plot::

    import fwiw.visualization
    fwiw.visualization.plot_domain(-20, +20, -20, +20, 3.0,
        rotation_axis=[1.0, 1.0, 1.0], rotation_angle_in_degree=-45.0,
        plot_simulation_domain=True)

Adding Seismic Events
---------------------
All events have to be stored in the *EVENTS* subfolder of the project. They
have to be valid QuakeML files with full moment tensor information. FWIW
provides some convenience methods for this purpose. One can leverage the IRIS
SPUD service (http://www.iris.edu/spud/momenttensor) to get GlobalCMT events.
Simply search for an event and copy the url. The **iris2quakeml** script will
then grab the QuakeML from the url and store an XML file in the current folder.

See :doc:`iris2quakeml` for more information.

.. code-block:: bash

    $ cd EVENTS
    $ iris2quakeml http://www.iris.edu/spud/momenttensor/959525
    $ iris2quakeml http://www.iris.edu/spud/momenttensor/995655

All events can be viewed with

.. code-block:: bash

    $ fwiw plot_events


.. plot::

    import fwiw.visualization
    map = fwiw.visualization.plot_domain(-20, +20, -20, +20, 3.0,
        rotation_axis=[1.0, 1.0, 1.0], rotation_angle_in_degree=-45.0,
        show_plot=False)
    # Create event.
    from obspy.core.event import *
    ev = Event()
    cat = Catalog(events=[ev])
    org = Origin()
    fm = FocalMechanism()
    mt = MomentTensor()
    t = Tensor()
    ev.origins.append(org)
    ev.focal_mechanisms.append(fm)
    fm.moment_tensor = mt
    mt.tensor = t
    org.latitude = 37.4
    org.longitude = -24.38
    t.m_rr = -1.69e+18
    t.m_tt = 9.12e+17
    t.m_pp = 7.77e+17
    t.m_rt = 8.4e+16
    t.m_rp = 2.4e+16
    t.m_tp = -4.73e+17
    ev2 = Event()
    cat.append(ev2)
    org = Origin()
    fm = FocalMechanism()
    mt = MomentTensor()
    t = Tensor()
    ev2.origins.append(org)
    ev2.focal_mechanisms.append(fm)
    fm.moment_tensor = mt
    mt.tensor = t
    org.latitude = 35.9
    org.longitude = -10.37
    t.m_rr = 6.29e+17
    t.m_tt = -1.12e+18
    t.m_pp = 4.88e+17
    t.m_rt = -2.8e+17
    t.m_rp = -5.22e+17
    t.m_tp = 3.4e+16
    fwiw.visualization.plot_events(cat, map)


Waveform Data
-------------
Every inversion needs real data to be able to quantify misfits. The waveform
data for all events are stored in the *DATA* subfolder. The data for each
single event will be stored in a subfolder of the *DATA* folder with the
**same name as the QuakeML file minus the .xml**.

To automatically create the necessary folder for each event run

.. code-block:: bash

    $ fwiw update_structure

This will result in a directory structure in the fashion of::

    MyInversion
    |-- DATA
    |   |-- GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
    |   |-- GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9
    |-- EVENTS
    |   |-- GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35.xml
    |   |-- GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9.xml
    |-- LOGS
    |-- MODELS
    |-- STATIONS
    |   |-- RESP
    |   |-- SEED
    |   |-- StationXML
    |-- SYNTHETICS
    |   |-- GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
    |   |-- GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9
    |-- config.xml


All data in the *DATA* subfolder has to be real data. The data is further
structured by assigning a tag to every data set. A tag is assigned by simply
placing a folder in *ROOT/DATA/EVENT_NAME* and putting all data in there. The
special tag *raw* is reserved for the raw waveforms straight from the
datacenters or some other source. Other tags should describe the filtering and
processing applied to the data. The same is true for synthetic waveform data,
except that in that case, the data resides in the *SYNTHETICS* folder and the
tags should describe the simulation ran to obtain the waveforms.

After a while, the structure might look like this::

    MyInversion
    |-- DATA
        |-- GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
            |-- raw
                ...
            |-- 100s_to_10s_bandpass
                ...
            |-- 200s_to_20s_bandpass
                ...
    |-- SYNTHETICS
        |-- GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
            |-- inversion_1_100s
                ...
            |-- inversion_2_100s
                ...
            |-- inversion_2_50s
                ...
    |-- ...

**The user is responsible** for adhering to that structure. Otherwise other
parts of FWIW cannot operate properly.

Station Data
------------
FWIW needs to know the coordinates and instrument response of each channel.
One way to achieve this to use SAC files, which contain coordinates, and RESP
files containing the response information for each channel. Another possibility
is to use MiniSEED waveform data and the corresponding dataless SEED or
StationXML files. Please keep in mind that FWIW currently expects to only
have channels of one station in each dataless SEED and StationXML file.

Naming scheme
^^^^^^^^^^^^^

**dataless SEED**

All dataless SEED files are expected to be in the *STATIONS/SEED* directory and
be named after the following scheme::

    dataless.NETWORK_STATION[.X]

*NETWORK*, and *STATION* should be replaced with the corresponding network and
stations codes. It is possible that multiple files are needed for each station
(e.g. different files for different time intervals/channels) and thus *.1*,
*.2*, ... can be appended to the filename. FWIW will automatically choose
the correct file in case they need to be accessed.

**StationXML**

All StationXML files are expected to be placed in the *STATIONS/StationXML*
folder and following the scheme::

    station.NETWORK_STATION[.X].xml

The logic for for the different parts is the same as for the dataless SEED
files described in the previous paragraph.

**RESP Files**

All RESP files are to be put in the *STATIONS/RESP* folder with the following
name::

    RESP.NETWORK.STATION.LOCATION.CHANNEL[.X]

In contrast to the two other station information formats the RESP filename also
has to include the location and channel identifiers.


Download Helpers
----------------

FWIW comes with a collection of scripts that help downloading waveform and
station data from the IRIS and ArcLink services. Waveform data will always be
downloaded as MiniSEED. Station data will, due to the different products of the
dataservices, either be downloaded as StationXML (IRIS) or dataless SEED.
Furthermore, as many tools so far are not able to deal with StationXML data,
the RESP files for each channel will also be downloaded. This is redundant
information but enables the use of many tools otherwise not possible.

Downloading Waveforms
^^^^^^^^^^^^^^^^^^^^^

Waveforms are downloaded on a per event basis. The **config.xml** file contains
some specification to detail the download. Each event is referred to by its
name which is simply the filename minus the extension. To get a list of all
events in the current project just execute

.. code-block:: bash

    $ fwiw list_events
    2 events in project:
        GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
        GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9


To download the waveform data for one event, choose one and run

.. code-block:: bash

    $ fwiw download_waveforms GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35


This, dependent on the domain size, event location, and origin time can take a
while. Executing the same command again will only attempt to download data not
already present. All data will be placed in `DATA/EVENT_NAME/raw`.


Downloading Station Data
^^^^^^^^^^^^^^^^^^^^^^^^

FWIW also includes some functionality to download station metadata. It will,
download StationXML and RESP files from IRIS and dataless SEED and RESP files
from ArcLink. It works the same as it does for the waveforms. To download all
stations for one event simply execute

.. code-block:: bash

    $ fwiw download_stations GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35

.. note::

    The `fwiw download_stations` command will, for the specified event, figure
    what waveform data is present in the `DATA/EVENT_NAME/raw` folder and
    download all missing station metadata information for these files.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
