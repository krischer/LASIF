.. LASIF documentation master file, created by
   sphinx-quickstart on Fri Feb  1 15:47:43 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to LASIF's Documentation!
===================================

.. image:: images/logo/lasif_logo.png
    :width: 50%

LASIF (**L**\ arg-scale **S**\ eismic **I**\ nversion **F**\ ramework) is a
**data-driven workflow tool** to perform full waveform inversions.  It is
opinionated and strict meaning that it enforces a certain data and directory
structure. The upside is that it only requires a very minimal amount of
configuration and maintenance. It attempts to gather all necessary information
from the data itself so there is no need to keep index or content files.

All parts of LASIF also work completely on their own; see the class and
function documentation at the end of this document. Furthermore LASIF offers a
project based inversion workflow management system which is introduced in the
following tutorial.

LASIF works with the notion of so called inversion projects. A project is
defined as a series of iterations working on the same physical domain. Where
possible and useful, LASIF will use XML files to store information. The
reasoning behind this is twofold. It is easily machine and human readable. It
furthermore also serves as a preparatory step towards a fully database driven
full waveform inversion workflow as all necessary information is already stored
in an easily indexable data format.

LASIF is data-driven meaning that it attempts to gather all necessary
information from the available data. The big advantage of this approach is that
the users can use any tool they want to access and work with the data as long
as they adhere to the directory structure imposed by LASIF. At the start of
every LASIF operation, the tool checks what data is available and uses it. To
achieve reasonable performance it employs a transparent caching scheme able to
quickly any changes the user makes to the data. Also important to keep in mind
is that **LASIF will never delete any data**.


TO DO: Things that are still missing
====================================

This is mainly useful for the developers.

* Applying the time corrections for events and stations
* Settings for the time frequency misfit
* A clean way to integrate other misfits
* Data rejection criteria implementation
* Log more things for better provenance


Supported Data Formats
======================

This is a short list of supported data formats and other software.


* **Waveform Data:** All file formats supported by ObsPy.
* **Synthetics:** All file formats supported by ObsPy and the output files of
  SES3D 4.0.
* **Event Metadata:** QuakeML 1.2
* **Station Metadata:** dataless SEED, RESP and (hopefully) soon FDSN
  StationXML.  Once implemented, StationXML will be the recommended and most
  future proof format.
* **Earth Models:** Currently the raw SES3D model format is supported.
* **Waveform Solvers:** SES3D 4.0, support for SpecFEM Cartesian and/or Globe
  will be added soon.


Further Notes
-------------

QuakeML files
^^^^^^^^^^^^^
LASIF is designed to work with valid QuakeML 1.2 event files. Please assure
that the files you use actually are just that. If possible try to only use
QuakeML files with one origin and one focal mechanism, otherwise LASIF will
choose the preferred origin and/or focal mechanism (or the first of each, if no
preferred one is specified). **The origin time specified in the QuakeML file
will be the reference time for each event!** Times specified in SAC files will
be ignored.

This also means that the raw data files have to have the correct time
information.

Synthetic Data Files
^^^^^^^^^^^^^^^^^^^^
The very first sample of each synthetic waveform will be assumed to coincide
with the event time. If this is not a reasonable assumption, please contact the
LASIF developers.


Tutorial
========
This tutorial will teach you how to perform an iterative full waveform
inversion with LASIF and SES3D by example.

The example used throughout this tutorial is the same as given in the SES3D
Documentation except that the used events differ. It is a good idea to also
have the SES3D documentation at hand.


Command Line Interface
----------------------

LASIF ships with a command line interface, consisting of a single command:
**lasif**.

Assuming the installation was successful, the following command will print a
short overview of all commands available within LASIF:

.. code-block:: bash

    $ lasif help

    Usage: lasif FUNCTION PARAMETERS

    Available functions:
        add_spud_event
        create_new_iteration
        ...

To learn more about a specific command, append *help* to it:

.. code-block:: bash

    $ lasif init_project help

    Usage: lasif init_project FOLDER_PATH

        Creates a new LASIF project at FOLDER_PATH. FOLDER_PATH must not exist
        yet and will be created.


.. note::

    All **lasif** commands work and use the correct project as long as they are
    executed somewhere inside a project's folder structure. It will recursively
    search the parent directories until it finds a *config.xml* file. This will
    then be assumed to be the root folder of the project.

Now that the preliminaries have been introduced, let's jump straight to the
example.

Creating a New Project
----------------------
The necessary first step, whether for starting a new inversion or migrating an
already existing inversion to LASIF, is to create a new project. In the
following the project will be called **TutorialAnatolia**.

.. code-block:: bash

    $ lasif init_project TutorialAnatolia

This will create the following directory structure. It will be explained in
more detail later on::

    TutorialAnatolia
    |-- ADJOINT_SOURCES_AND_WINDOWS
    |-- CACHE
    |-- config.xml
    |-- DATA
    |-- EVENTS
    |-- ITERATIONS
    |-- LOGS
    |-- MODELS
    |-- OUTPUT
    |-- STATIONS
    |   |-- RESP
    |   |-- SEED
    |   |-- StationXML
    |-- SYNTHETICS


The configuration for each project is defined in the **config.xml** file. It is
a simple, self-explanatory XML format. After the project has been initialized
it will look akin to the following:

.. code-block:: xml

    <?xml version="1.0" encoding="utf-8"?>
    <lasif_project>
        <name>TutorialAnatolia</name>
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
    </lasif_project>

It should be fairly self-explanatory.

* *name* denotes a short description of the project. Usually the same as the
  folder name.
* *description* can be any further useful information about the project. This
  is not used by LASIF but potentially useful for yourself.
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
* The *domain* settings will be explained in more detail in the following
  paragraphs.
* The *boundary_width_in_degree* tag is use to be able to take care of the
  boundary conditions, e.g. data will be downloaded within
  *boundary_width_in_degree* distance to the domain border.

The file, amongst other settings, defines the physical domain for the
inversion. Please set it to the following (same as in the SES3D Tutorial):

* Latitude: **34.1° - 42.9°**
* Longitude: **23.1° - 42.9°**
* Depth: **0 km - 471 km**
* Boundary width in degree: **1.46°**

In generally one should only work with data not affected by the boundary
conditions. SES3D utilizes perfectly matched layers boundary conditions (PML).
It is not advisable to use data that traverses these layers. The default
setting of SES3D is to use two boundary layers. In this example this amounts to
(in longitudinal direction) 1.46°. In a real world case it is best to use some
more buffer layers to avoid boundary effects. In this small example this would
influence the domain too much so we just set it to 1.46°.

At any point you can have a look at the defined domain with

.. code-block:: bash

    $ lasif plot_domain

This will open a window showing the location of the physical domain and the
simulation domain. The inner contour shows the domain minus the previously
defined boundary width.

.. plot::

    import lasif.visualization
    lasif.visualization.plot_domain(34.1, 42.9, 23.1, 42.9, 1.46,
        rotation_axis=[0.0, 0.0, 1.0], rotation_angle_in_degree=0.0,
        plot_simulation_domain=True, zoom=True)


The nature of SES3D's coordinate system has the effect that simulation is most
efficient in equatorial regions. Thus it is oftentimes advantageous to rotate
the frame of reference so that the simulation happens close to the equator.
LASIF first defines the simulation domain; the actual simulation happens here.
Optional rotation parameters define the physical location of the domain. The
coordinate system for the rotation parameters is described in
:py:mod:`lasif.rotations`.  You will have to edit the file to adjust it to your
region of interest. The rotation functionality is not used in this Tutorial's
example; in case it is used, simulation and physical domain would differ.
LASIF handles all rotations necessary so the user never needs to worry about
these. Just keep in mind to always keep any data (real waveforms, station
metadata and events) in coordinates that correspond to the physical domain and
all synthetic waveforms in coordinates that correspond to the simulation
domain. If the domain is rotated, the **plot_domain** command will plot both,
the physical and the simulation domain:

.. plot::

    import lasif.visualization
    lasif.visualization.plot_domain(-20, +20, -20, +20, 3.0,
        rotation_axis=[1.0, 1.0, 1.0], rotation_angle_in_degree=-45.0,
        plot_simulation_domain=True)

.. note::

    The map projection and zoom should automatically adjust so it is suitable
    for the dimensions and location of the chosen domain. If that is not the
    case please file an issue on the project's Github page.

The small size of the domain does not warrant downloading an hour worth of data
for every event. Half an hour or event less is more then sufficient. After all
the discussed changes the **config.xml** file should be similar to this one:

.. code-block:: xml

    <?xml version='1.0' encoding='UTF-8'?>
    <lasif_project>
      <name>TutorialAnatolia</name>
      <description>Tutorial Inversion</description>
      <download_settings>
        <arclink_username>your@email.com</arclink_username>
        <seconds_before_event>300</seconds_before_event>
        <seconds_after_event>1800</seconds_after_event>
      </download_settings>
      <domain>
        <domain_bounds>
          <minimum_longitude>23.1</minimum_longitude>
          <maximum_longitude>42.9</maximum_longitude>
          <minimum_latitude>34.1</minimum_latitude>
          <maximum_latitude>42.9</maximum_latitude>
          <minimum_depth_in_km>0.0</minimum_depth_in_km>
          <maximum_depth_in_km>471.0</maximum_depth_in_km>
          <boundary_width_in_degree>1.46</boundary_width_in_degree>
        </domain_bounds>
        <domain_rotation>
          <rotation_axis_x>0.0</rotation_axis_x>
          <rotation_axis_y>0.0</rotation_axis_y>
          <rotation_axis_z>1.0</rotation_axis_z>
          <rotation_angle_in_degree>0.0</rotation_angle_in_degree>
        </domain_rotation>
      </domain>
    </lasif_project>

This concludes the intial setup part of the inversion.


Adding Seismic Events
---------------------
Once the domain has been adjusted to your needs, you need to tell LASIF which
events you want to use for the inversion. This works by simply placing a valid
QuakeML 1.2 file at the correct location.

All events have to be stored in the *EVENTS* subfolder of the project. They
have to be QuakeML 1.2 files with full moment tensor information.

LASIF provides some convenience methods for this purpose. One can leverage the
IRIS SPUD service (http://www.iris.edu/spud/momenttensor) to get GlobalCMT
events.  Simply search for an event and copy the url. The **iris2quakeml**
script will then grab the QuakeML from the url and store an XML file in the
correct folder.

See :doc:`iris2quakeml` for more information. The LASIF command lines tools
contain a convenience wrapper around it that also makes sure that the events
ends up in the correct folder and gives them a reasonable filename that should
ease event identification.

.. code-block:: bash

    $ lasif add_spud_event http://www.iris.edu/spud/momenttensor/735711
    $ lasif add_spud_event http://www.iris.edu/spud/momenttensor/988455


These two commands should create two QuakeML files. To which events are
currently defined in the project use the **list_events** command.

.. code-block:: bash

    $ lasif list_events

    2 events in project:
        GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15
        GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11

You will notice that events are identified via their filename minus the
extension. This is an easy and flexible solution enabling you to tag the events
as you see fit. The slight disadvantage of this approach is that **you must not
change the event filenames after you have worked with them** because all
additional information for that event will be related to it via the event
filename. So please give them a good and reasonable filename. If you really
feel that event renaming is a necessary feature please file an issue on Github
so that the authors can add a proper event renaming function.

The **plot_events** command will return a map with all events currently part of
the project.

.. code-block:: bash

    $ lasif plot_events

.. plot::

    import lasif.visualization
    map = lasif.visualization.plot_domain(34.1, 42.9, 23.1, 42.9, 1.46,
        rotation_axis=[0.0, 0.0, 1.0], rotation_angle_in_degree=0.0,
        show_plot=False, zoom=True)
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
    org.latitude = 39.15
    org.longitude = 29.1
    t.m_rr = -8.07e+17
    t.m_tt = 8.92e+17
    t.m_pp = -8.5e+16
    t.m_rt = 2.8e+16
    t.m_rp = -5.3e+16
    t.m_tp = -2.17e+17
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
    org.latitude = 38.82
    org.longitude = 40.14
    t.m_rr = 5.47e+15
    t.m_tt = -4.11e+16
    t.m_pp = 3.56e+16
    t.m_rt = 2.26e+16
    t.m_rp = -2.25e+16
    t.m_tp = 1.92e+16
    lasif.visualization.plot_events(cat, map)


The **event_info** command is your friend if you desire more information about
a certain event:

.. code-block:: bash

    $ lasif event_info GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11

    Earthquake with 5.1 Mwc at TURKEY
        Latitude: 38.820, Longitude: 40.140, Depth: 4.5 km
        2010-03-24T14:11:31.000000Z UTC

    Station and waveform information available at 0 stations:

    ===========================================================================
                 id       latitude      longitude      elevation    local depth
    ===========================================================================

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


Adding Waveform Data
--------------------
Every inversion needs real data to be able to quantify misfits. The waveform
data for all events are stored in the *DATA* subfolder. The data for each
single event will be stored in a subfolder of the *DATA* folder with the
**same name as the QuakeML file minus the .xml**.

These folder are automatically created and updated each time a lasif command is
executed. If you followed the tutorial, your directory structure should
resemble the following::

    TutorialAnatolia
    |── ADJOINT_SOURCES_AND_WINDOWS
    |   |── ADJOINT_SOURCES
    |   |── WINDOWS
    |── CACHE
    |── config.xml
    |── DATA
    |   |── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11
    |   |── GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15
    |── EVENTS
    |   |── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11.xml
    |   |── GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15.xml
    |── ITERATIONS
    |── LOGS
    |── MODELS
    |── OUTPUT
    |── STATIONS
    |   |── RESP
    |   |── SEED
    |   |── StationXML
    |── SYNTHETICS
        |── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11
        |── GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15


All data in the *DATA* subfolder has to be processed or unprocessed actual
data. The data is further structured by assigning a tag to every data set. A
tag is assigned by simply placing a folder in *ROOT/DATA/EVENT_NAME* and
putting all data in there. The special tag *raw* is reserved for the raw
waveforms straight from the datacenters or some other source. Other tags should
describe the filtering and processing applied to the data (LASIF's built-in
processing capabilities actually enforce certain tag names - this will be
covered in more detail later on). The same is true for synthetic waveform data,
except that in that case, the data resides in the *SYNTHETICS* folder and the
tags have to coincide with the iteration names. More on this later on.

After a while, the structure might look like this::

    TutorialAnatolia
    |-- DATA
        |── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11
            |-- raw
            |-- preprocessed_hp_0.01000_lp_0.12500_npts_4000_dt_0.130000
        |...
    |-- SYNTHETICS
        |── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11
            |-- inversion_1
            |-- inversion_2
            |...
        |...
    |...

**The user is responsible** for adhering to that structure. Otherwise other
parts of LASIF cannot operate properly. Many commands shipping with LASIF ease
that process.

Station Data
------------
LASIF needs to know the coordinates and instrument response of each channel.
One way to achieve this to use SAC files, which contain coordinates, and RESP
files containing the response information for each channel. Another possibility
is to use MiniSEED waveform data and the corresponding dataless SEED or
StationXML files. Please keep in mind that LASIF currently expects to only
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
*.2*, ... can be appended to the filename. LASIF will automatically choose
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

LASIF comes with a collection of scripts that help downloading waveform and
station data from the IRIS and ArcLink services. Waveform data will always be
downloaded as MiniSEED. Station data will, due to the different products of the
dataservices, either be downloaded as StationXML (IRIS) or dataless SEED.
Furthermore, as many tools so far are not able to deal with StationXML data,
the RESP files for each channel will also be downloaded. This is redundant
information but enables the use of many tools otherwise not possible.

Downloading Waveforms
^^^^^^^^^^^^^^^^^^^^^

Waveforms are downloaded on a per event basis. The **config.xml** file contains
some specification to detail the download.

To download the waveform data for one event, choose one and run

.. code-block:: bash

    $ lasif download_waveforms GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11


The command essentially just tries to download everything it can. It queries
the IRIS DMC and ArcLink for all stations available in the physical domain and
then downloads the appropriate data. It accounts for the domain borders and
possible domain rotations. It is influences by three parameters in the
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

This, dependent on the domain size, event location, and origin time can take a
while. Executing the same command again will only attempt to download data not
already present. All data will be placed in `DATA/EVENT_NAME/raw`.

.. note::

    At this point it is worth mentioning that LASIF keeps logs of many actions
    that the user performs. All logs will be saved in the *LOGS* subfolder.


Downloading Station Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LASIF also includes some functionality to download station metadata. It will,
download RESP files from IRIS and dataless SEED files from ArcLink. It works
the same as it does for the waveforms. To download all stations for one event
simply execute

.. code-block:: bash

    $ lasif download_stations GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11

The `lasif download_stations` command will, for the specified event, figure
what waveform data is present in the `DATA/EVENT_NAME/raw` folder and download
all missing station metadata information for these files.

.. note::

    At some point in the near future the station metadata downloading routines
    will be changed so that they exclusively work with StationXML metadata.


Inspecting the Data
-------------------

Once waveform and station metadata has been downloaded (either with the
built-in helpers or manually) and placed in the correct folders, LASIF can
start to work with it.

.. note::

    LASIF essentially needs three ingredients to be able to interpret waveform
    data:

    * The actual waveforms
    * The location of the recording seismometer
    * The instrument response for each channel at the time of data recording

    Some possibilities exist to specify these:

    * MiniSEED data and dataless SEED for the metadata (currently preferred)
    * SAC data and RESP files (needed for legacy reasons)
    * MiniSEED and RESP files (this combination does not actually contain
      location information but LASIF launches some web requests to get just the
      locations and stores them in a cache database)
    * Most other combinations should also work but have not been tested.

    In the future the preferred way will be miniSEED data combined with
    StationXML metadata. This provides a clear seperation of data and metadata.


At this point, LASIF is able to match available station and waveform
information. Only stations where the three aforementioned ingredients are
available will be considered to be stations that are good to be worked with by
LASIF. Others will be ignored.

To get an overview, of what data is actually available for any given event,
just execute:

.. code-block:: bash

    $ lasif event_info GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11

    Earthquake with 5.1 Mwc at TURKEY
        Latitude: 38.820, Longitude: 40.140, Depth: 4.5 km
        2010-03-24T14:11:31.000000Z UTC

    Station and waveform information available at 8 stations:

    ===========================================================================
                 id       latitude      longitude      elevation    local depth
    ===========================================================================
             GE.APE        37.0689        25.5306          620.0            0.0
             GE.ISP        37.8433        30.5093         1100.0            5.0
             HL.APE        37.0689        25.5306          620.0            0.0
             HL.ARG         36.216         28.126          170.0            0.0
             HL.RDO         41.146         25.538          100.0            0.0
             HT.ALN        40.8957        26.0497          110.0            0.0
            HT.SIGR        39.2114        25.8553           93.0            0.0
            IU.ANTO         39.868        32.7934         1090.0           None


.. note::

    As seen here, the local depth can is allowed to not be set. In this cases
    it will be assumed to be zero. For all practical purposes the local depth
    does not matter for continental scale inversions.


It is furthermore possible to plot the availability information for one event
including a very simple ray coverage plot with:

.. code-block:: bash

    $ lasif plot_event GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11

.. plot::

    import matplotlib.pylab as plt
    from obspy import UTCDateTime
    import lasif.visualization
    map = lasif.visualization.plot_domain(34.1, 42.9, 23.1, 42.9, 1.46,
        rotation_axis=[0.0, 0.0, 1.0], rotation_angle_in_degree=0.0,
        show_plot=False, zoom=True)
    event_info = {'depth_in_km': 4.5, 'region': 'TURKEY', 'longitude': 40.14,
        'magnitude': 5.1, 'magnitude_type': 'Mwc', 'latitude': 38.82,
        'origin_time': UTCDateTime(2010, 3, 24, 14, 11, 31)}
    stations = {u'GE.APE': {'latitude': 37.0689, 'local_depth': 0.0,
        'elevation': 620.0, 'longitude': 25.5306}, u'HL.ARG': {'latitude':
        36.216, 'local_depth': 0.0, 'elevation': 170.0, 'longitude': 28.126},
        u'IU.ANTO': {'latitude': 39.868, 'local_depth': None, 'elevation':
        1090.0, 'longitude': 32.7934}, u'GE.ISP': {'latitude': 37.8433,
        'local_depth': 5.0, 'elevation': 1100.0, 'longitude': 30.5093},
        u'HL.RDO': {'latitude': 41.146, 'local_depth': 0.0, 'elevation': 100.0,
        'longitude': 25.538}, u'HT.SIGR': {'latitude': 39.2114, 'local_depth':
        0.0, 'elevation': 93.0, 'longitude': 25.8553}, u'HT.ALN': {'latitude':
        40.8957, 'local_depth': 0.0, 'elevation': 110.0, 'longitude': 26.0497},
        u'HL.APE': {'latitude': 37.0689, 'local_depth': 0.0, 'elevation':
        620.0, 'longitude': 25.5306}}
    lasif.visualization.plot_stations_for_event(map_object=map,
        station_dict=stations, event_info=event_info)
    # Create event.
    from obspy.core.event import *
    cat = Catalog(events=[])
    ev = Event()
    cat.append(ev)
    org = Origin()
    fm = FocalMechanism()
    mt = MomentTensor()
    t = Tensor()
    ev.origins.append(org)
    ev.focal_mechanisms.append(fm)
    fm.moment_tensor = mt
    mt.tensor = t
    org.latitude = 38.82
    org.longitude = 40.14
    t.m_rr = 5.47e+15
    t.m_tt = -4.11e+16
    t.m_pp = 3.56e+16
    t.m_rt = 2.26e+16
    t.m_rp = -2.25e+16
    t.m_tp = 1.92e+16
    lasif.visualization.plot_events(cat, map)
    plt.show()


If you are interested in getting a coverage plot of all events and data
available for the current project, please execute the **plot_raydensity**
command:

.. code-block:: bash

    $ lasif plot_raydensity

Keep in mind that this only results in a reasonable plot for large amounts of
data; for the toy example used in the tutorial it will not work. It is not a
physically accurate plot but helps in judging data coverage and directionality
effects. An example from a larger LASIF project illustrates this:


.. image:: images/raydensity.jpg
    :width: 70%
    :align: center



Interlude: Validating the data
------------------------------

You might have noticed that LASIF projects can potentially contain many million
files and it will thus be impossible to validate the data by hand. Therefore
LASIF contains a number of functions attempting to check the data of a
project. All of these can be called with

.. code-block:: bash

    $ lasif validate_data

Please make sure no errors appear otherwise LASIF cannot guarantee to work
correctly.  With time more checks will be added to this function as more
problems arise.


Defining a New Iteration
------------------------

LASIF organizes the actual inversion in an arbitrary number of iterations; each
of which is described by a single XML file. Within each file, the events and
stations for this iterations, the solver settings, and other information is
specified. Each iteration can have an arbitrary name. It is probably a good
idea to give simple numeric names, like 1, 2, 3, ...

Let's start by creating the XML file for the very first iteration with the
**create_new_iteration** command.

.. code-block:: bash

    $ lasif create_new_iteration 1 SES3D_4_0


This command takes two arguements; the first being the iteration name. A simple
number is sufficient in many cases. The second argument is the waveform solver
to be used for this iteration. It currently only supports SES3D 4.0 but the
infrastructure to add other solvers is already in place.

You will see that this create a new file; *ITERATIONS/ITERATION_1.xml**. Each
iteration will have its own file. To get a list of iterations, use

.. code-block:: bash

    $ lasif list_iterations

    1 Iteration in project:
        1


To get more information about a specific iteration, use the **iteration_info** command.

.. code-block:: bash

    $ lasif iteration_info 1

    LASIF Iteration
        Name: 1
        Description: None
        Source Time Function: Filtered Heaviside
        Preprocessing Settings:
                Highpass Period: 100.000 s
                Lowpass Period: 8.000 s
        Solver: SES3D 4.0 | 500 timesteps (dt: 0.75s)
        2 events recorded at 10 unique stations
        16 event-station pairs ("rays")

.. note::

    You might have noticed the pairs of **list_x** and **x_info** commands, e.g.
    **list_events** and **event_info** or **list_iterations** and
    **iteration_info**. This scheme is true for most things in LASIF. The
    **list_x** variant is always used to get a quick overview of everything
    currently part of the LASIF project. The **x_info** counterpart returns
    more detailed information about the resource.

The Iteration XML Files
^^^^^^^^^^^^^^^^^^^^^^^

The XML file defining each iteration attempts to be a collection of all
information relevant for a single iteration.

.. note::

    The iteration XML files are the **main provenance information** (in
    combination with the log files) within LASIF. By keeping track of what
    happened during each iteration it is possible to reasonably judge how any
    model came into being.

    If at any point you feel the need to keep track of additional information
    and there is no place for it within LASIF, please contact the developers.
    LASIF aims to offer an environment where all necessary information can be
    stored in an organized and sane manner.


The iteration XML files currently contain:

* Some metadata: the iteration name, a description and some comments.
* A limited data preprocessing configuration. The data preprocessing is
  currently mostly fixed and only the desired frequency content can be chosen.
  Keep in mind that these values will also be used to filter the source time
  function.
* Some data rejection criterias. This will be covered in more detail later on.
* The source time function configuration.
* The settings for the solver used for this iteration.
* A list of all events used for the iteration. Here it is possible to apply
  weight the different events and also to apply a time correction. It can be
  different per iteration.
* Each event contains a list of stations where data is available. Furthermore
  each station can have a different weight and time correction.

This file is rather verbose but also very flexible. It is usually only
necessary to create this file once and then make a copy and small adjustments
for each iteration. In the future some more user-friendly ways to deal with the
information will hopefully be incorporated into LASIF.


Let's have a quick look at the generated file. The **create_new_iteration**
command will create a new iteration file with all the information currently
present in the LASIF project.

.. code-block:: xml

    <?xml version='1.0' encoding='UTF-8'?>
    <iteration>
      <iteration_name>1</iteration_name>
      <iteration_description>The first iteration</iteration_description>
      <comment>This is just for the dummy tutorial example</comment>
      <comment>There can be an arbitrary number of comments</comment>
      <data_preprocessing>
        <highpass_period>100.0</highpass_period>
        <lowpass_period>8.0</lowpass_period>
      </data_preprocessing>
      <rejection_criteria>
        ...
      </rejection_criteria>
      <source_time_function>Filtered Heaviside</source_time_function>
      <solver_parameters>
        <solver>SES3D 4.0</solver>
        <solver_settings>
          <simulation_parameters>
            <number_of_time_steps>4000</number_of_time_steps>
            <time_increment>0.13</time_increment>
            <is_dissipative>false</is_dissipative>
          </simulation_parameters>
          <output_directory>../OUTPUT/CHANGE_ME/{{EVENT_NAME}}</output_directory>
          ...
        </solver_settings>
      </solver_parameters>
      <event>
        <event_name>GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15</event_name>
        <event_weight>1.0</event_weight>
        <time_correction_in_s>0.0</time_correction_in_s>
        <station>
          <station_id>HL.ARG</station_id>
          <station_weight>1.0</station_weight>
          <time_correction_in_s>0.0</time_correction_in_s>
        </station>
        <station>
          <station_id>IU.ANTO</station_id>
          <station_weight>1.0</station_weight>
          <time_correction_in_s>0.0</time_correction_in_s>
        </station>
        ...
      </event>
      <event>
        ...
      </event>
      ...
    </iteration>

It is a rather self-explaining file; some things to look out for:

* The dataprocessing frequency limits are given periods in seconds. This is
  more in line what one would normally use.
* The source time function is just given as a string. The "Filtered Heaviside"
  is the only source time function currently supported. It will be filtered
  with the limits specified in the data preprocessing section.
* The paths in the solver settings contains an **{{EVENT_NAME}}** part. This
  part will be replaced by the actual event name. This means that the file does
  not have to be adjusted for every event.

The file shown here has already be adjusted to be consistent with the SES3D
example. Please do the same here. Notably you have to adjust the number of time
steps and the time increment. Furthermore the paths have to be adjusted so that
they for the system you plan to run the simulations on.

Source Time Functions
^^^^^^^^^^^^^^^^^^^^^

The source time functions will be dynamically generated from the information
specified in the iteration XML files. Currently only one type of source time
function, a filtered Heaviside function is supported. In the future, if
desired, it could also be possible to use inverted source time functions.

The source time function will always be defined for the number of time steps
and time increment you specify in the solver settings. Furthermore all source
time functions will be filtered with the same bandpass as the data.

To get a quick look of the source time function for any given iteration, use
the **plot_stf** command with the iteration name:

.. code-block:: bash

    $ lasif plot_stf 1

This command will read the corresponding iteration file and open a plot with a
time series and a time frequency representation of the source time function.

.. plot::

    import lasif.visualization
    from lasif.source_time_functions import filtered_heaviside

    data = filtered_heaviside(4000, 0.13, 1.0 / 500.0, 1.0 / 60.0)
    lasif.visualization.plot_tf(data, 0.13)

Data Preprocessing
^^^^^^^^^^^^^^^^^^

Data preprocessing is an essential step if one wants to compare data and
seismograms. It serves several purposes: Restricting the frequency content of
the data to that of the synthetics - what is not simulated can no be seen in
synthetic seismograms. Remove the instrument response and convert to the same
units used for the synthetics (usually m\s). Furthermore any linear trends and
static offset are removed and the some processing has to be performed so that
the data is available at the same points in time as the synthetics. The goal of
the preprocessing within LASIF is to create data that is directly comparable to
simulated data without any more processing.

The applied processing is identified via the folder name::

    preprocessed_hp_0.01000_lp_0.12500_npts_4000_dt_0.130000

or (in Python terms):

.. code-block:: python

    highpass = 1.0 / 100.0
    lowpass = 1.0 / 8.0
    npts = 4000
    dt = 0.13

    processing_tag = ("preprocessed_hp_{highpass:.5f}_lp_{lowpass:.5f}_"
        "npts_{npts}_dt_{dt:5f}").format(highpass=highpass, lowpass=lowpass,
        npts=npts, dt=dt)

.. note::

    You can use any processing tool you want, but you have to adhere to the
    directory structure otherwise LASIF will not be able to work with the data.
    It is furthermore important that the processed filenames are identical to
    the unprocessed ones.

    If you feel that additional identifiers are needed to uniquely identify the
    applied processing (in the limited setting of being useful for the here
    performed full waveform inversion) please contact the LASIF developers.

You can of course also simply utilize LASIF's built-in preprocessing. Using it
is trivial, just launch the **preprocess_data** command together with the
iteration name.

.. code-block:: bash

    $ lasif preprocess_data 1

This will start a fully parallelized preprocessing for all data required for
the specified iteration. It will utilize all your machine's cores and might
take a while. If you repeat the command it will only process data not already
processed; an advantages is that you can cancel the processing at any time and
then later on just execute the command again to continue where you left off.
This usually only needs to be done every couple of iterations when you decide
to go to higher frequencies or add new data.

The preprocessed data will be put in the correct folder.

Data Rejection
^^^^^^^^^^^^^^

Coming soon...watch this space.


This concludes the initial setup for each iteration. The next steps is to
actually simulate anything and LASIF of course also assists in that regard.


Generating SES3D Input Files
----------------------------

LASIF is currently capable of producing input files for SES3D 4.0. It is very
straightforward and knows what data is available for every event and thus can
generate these files fully automatically. In the future it might be worth
investigating automatic job submission to high performance machines as this is
essentially just repetitive and error-prone work.

The iteration XML file also governs the solver used and the specific settings
used for the given iteration, e.g. the settings for the SES3D 4.0 solver are
shown here.

.. code-block:: xml

  <solver_parameters>
    <solver>SES3D 4.0</solver>
    <solver_settings>
      <simulation_parameters>
        <number_of_time_steps>4000</number_of_time_steps>
        <time_increment>0.13</time_increment>
        <is_dissipative>false</is_dissipative>
      </simulation_parameters>
      <output_directory>../OUTPUT/CHANGE_ME/{{EVENT_NAME}}</output_directory>
      <adjoint_output_parameters>
        <sampling_rate_of_forward_field>10</sampling_rate_of_forward_field>
        <forward_field_output_directory>
            ../OUTPUT/CHANGE_ME/ADJOINT/{{EVENT_NAME}}
        </forward_field_output_directory>
      </adjoint_output_parameters>
      <computational_setup>
        <nx_global>66</nx_global>
        <ny_global>108</ny_global>
        <nz_global>28</nz_global>
        <lagrange_polynomial_degree>4</lagrange_polynomial_degree>
        <px_processors_in_theta_direction>3</px_processors_in_theta_direction>
        <py_processors_in_phi_direction>4</py_processors_in_phi_direction>
        <pz_processors_in_r_direction>4</pz_processors_in_r_direction>
      </computational_setup>
    </solver_settings>
  </solver_parameters>

Most things should be self-explanatory.  In case something is not fully clear,
please refer to the SES3D 4.0 manual or contact the author. As previously
mentioned the **{{EVENT_NAME}}** placeholder will be replaced with the actual
event. Please take care that what you put in here it correct, otherwise the
simulations will not work. The settings shown here coincide with the settings
used in the SES3D 4.0 tutorial so we will just use those here.


Input File Generation
^^^^^^^^^^^^^^^^^^^^^

The actual input file generation is now very straightforward:


.. code-block:: bash

    $ lasif generate_input_files ITERATION_NAME EVENT_NAME SIMULATION_TYPE

**TYPE** has to be one of

    * *normal_simulation* - Use this if you want to get some waveforms.
    * *adjoint_forward* - Use this for the forward adjoint simulation. Please
      note that it requires a huge amount of disk space for the forward
      wavefield.
    * *adjoint_reverse* - Use this for the adjoint simulation. This requires
      that the adjoint sources have already been calculated. More on that later
      on.

The other parameters should be clear.

For this tutorial you can generate input files for both events with

.. code-block:: bash

    $ lasif generate_input_files 1 GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11 adjoint_forward
    $ lasif generate_input_files 1 GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15 adjoint_forward


This will place input files in the *OUTPUT* subdirectory of the project. In
general it is advisable to never delete the input files to facilitate
provenance and reproducibility.

If you are working in a rotated domain, all station coordinates and moment
tensors will automatically be rotated accordingly so that the actual simulation
can take place in an unrotated frame of reference.

Together with some models, these file can directly be used to run SES3D. For
the first couple of runs it is likely a good idea to check these file by hand
to verify your setup and potentially also the correctness of this tool suite.


Organizing the Models
---------------------

Short Deviation: Creating an initial model with SES3D 4.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is very quick tutorial to help you get up to speed with the model
generation for this tutorial with SES3D 4.0. You will still need to read the
SES3D manual. This part assumes that you created the input files according to
the previous section and that you have a copy of SES3D on the supercomputer.
Please not that you will have to adjust settings if you did not follow along
with the tutorial. The following has to take place on a machine with at least
48 available CPU cores.

1. Copy all generated input files to *SES3D/INPUT*.
2. Edit *nx_max*, *ny_max*, *nz_max* in *SES3D/SOURCE/ses3d_modules.f90*
3. Compile SES3D and model tools (execute *s_make* in *SES3D/MODELS/MODELS* and
   *SOURCE*).
4. Generate a homogeneous model by launching
   *SES3D/MODELS/MAIN/generate_models.exe* with 48 cores.
5. Now add some perturbations to get a 3D model by running
   *SES3D/MODELS/MAIN/add_perturbation.exe* with 48 cores.

You should now have lots of files in *SES3D/MODELS/MODELS*. These represent the
model in a format SES3D can use.

Models in LASIF
^^^^^^^^^^^^^^^

LASIF can directly deal with the models used in SES3D. Each model has to placed
in a subfolder of *MODELS*. The folder name will again be used to identify the
model. For this tutorial, place the just created files in
*MODELS/Intial_Model*.

Now you are able to use the **list_models** commands.

.. code-block:: bash

    lasif list_models

        1 model in project:
            Initial_Model

LASIF has some functionality to view the models. To launch the model viewer use
the **plot_model** command together with the model name.

.. code-block:: bash

    lasif plot_model Initial_Model

    Raw SES3D Model (split in 48 parts)
        Setup:
                Latitude: 34.10 - 42.90
                Longitude: 23.10 - 42.90
                Depth in km: 0.00 - 471.00
                Total element count: 211787
                Total grid point count: 13753701
        Memory requirement per component: 52.5 MB
        Available components: A, B, C, lambda, mu, rhoinv
        Available derived components: rho, vp, vsh, vsv
        Parsed components:

    Enter 'COMPONENT DEPTH' ('quit' to exit):


This will print some information about the model like the available components
and the components it can derive from these. Keep in mind that for plotting one
of the derived components it potentially has to load two or more components so
keep an eye on your machines memory. The tool can currently plot horizontal
slices for arbitrary components at arbitrary depths. To do this, simply type
the component name and the desired depth in kilometer and hit enter. This opens
a new window, e.g. for **vsv 100**:

.. image:: images/vsv_100km.jpg
    :width: 90%
    :align: center

Clicking at any point of the model pops up a vertical profile of the chosen
component at the clicked position. Closing the window again will enable you to
plot a different component or different depth. To leave the model viewer simply
type **quit**.


Synthetics
----------

Now that everything is set up, you have to actually perform the simulations.
Please keep in mind that the adjoint forward simulation require a very large
amount of disc space due to the need to store the forward wavefield. **The
example for this tutorial requires around 450 GB.**

The important output of the simulation are the waveform files. These should be
placed in the *SYNTHETICS/{{EVENT_NAME}}/ITERATION_{{ITERATION_NAME}}* folder.
So for the given examples, they should be placed in the
*SYNTHETICS/GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11/ITERATION_1* and
*SYNTHETICS/GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15/ITERATION_1*. Just put
the raw output files of the simulation in the corresponding folder; there is
no need to process them in any way.


Misfit and Adjoint Source Calculation
-------------------------------------

In order to simulate the adjoint wavefield one needs to calculate the adjoint
sources. An adjoint source is usually dependent on the misfit between the
synthetics and real data.

LASIF currently supports misfits in the time-frequency domain as defined by
Fichtner, 2008. Great care has to be taken to avoid cycle skips/phase jumps
between synthetics and data. This is achieved by careful windowing.

Weighting Scheme
^^^^^^^^^^^^^^^^

You might have noticed that at various points it has been possible to
distribute weights. These weights all contribute to the final adjoint source.
The inversion scheme requires one adjoint source per iteration, event, station,
and component.

In each iteration's XML file you can specify the weight for each event, denoted
as :math:`w_{event}` in the following. The weight is allowed to range between
0.0 and 1.0. A weight of 0.0 corresponds to no weight at all, e.g. skip the
event and a weight of 1.0 is the maximum weight.

Within each event is possible to weight each station separately, in the
following named :math:`w_{station}`. The station weights can also range from
0.0 to 1.0 and follow the same logic as the event weights.

You can furthermore choose an arbitrary number of windows per component for
which the misfit and adjoint source will be calculated. Each window has a
separate weight with the only limitation being that the weight has to be
positive. Assuming :math:`N` windows in a given component, the corresponding
adjoint sources will be called :math:`ad\_src_{1..N}` while their weights are
:math:`w_{1..N}`.

The final adjoint source for every component will be calculated according to
the following formula:

.. math::

   adj\_source = w_{event} \cdot w_{station} \cdot \frac{1}{\sum_{i=1}^N w_i} \cdot \sum_{i=1}^N w_i \cdot ad\_src_i

Misfit GUI
^^^^^^^^^^

LASIF comes with a graphical utility dubbed the Misfit GUI, that helps to pick
correct windows.

To launch it, execute the **launch_misfit_gui** together with the iteration
name and the event name.

.. code-block:: bash

    $ lasif launch_misfit_gui 1 GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11


This will open a window akin to the following:

.. image:: images/misfit_gui.png
    :width: 90%
    :align: center

It is essentially partitioned into three parts. The top part is devoted to plot
all three traces of a single station. The bottom left part shows a
representation of the misfit for the last chosen window and the bottom left
part shows a simple map.

.. note::

    The current interface is based purely on matplotlib. This has the advantage
    of keeping dependencies to minimum. Unfortunately matplotlib is not a GUI
    toolkit and therefore the GUI is not particularly pleasing from a UI/UX
    point of view. Some operations might feel clunky. We might move to a proper
    GUI toolkit in the future.

With the **Next** and **Prev** button you can jump from one station to the
next. The **Reset Station** button will remove all windows for the current
station.

The weight for any window has to be chosen before the windows are picked. To
chose the current weight, press the **w** key. At this point, the weight box
will be red. Now simply type the desired new weight and press **Enter** to
finish setting the new weight. All windows chosen from this point on will
be assigned this weight.

To actually choose a window simply drag in any of the waveform windows. Upon
mouse button release the window will be saved and the adjoint source will be
calculated. The number in the top left of each chosen window reflects the
weight for that window.

Right clicking on an already existing window will delete it, left clicking will
plot the misfit once again.

.. note::

    At any point you can press **h** to get an up-to-date help text for the
    GUI.


Final Adjoint Source Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During window selection the adjoint source for each chosen window will be
stored separately. To combine them to, apply the weighting scheme and convert
them to a format, that SES3D can actually use, run the
**finalize_adjoint_sources** command with the iteration name and the event
name.

.. code-block:: bash

    $ lasif finalize_adjoint_sources 1 GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11

This will also rotate the adjoint sources to the frame of reference used in the
simulations.

If you pick any more windows or change them in any way, you need to run the
command again.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
