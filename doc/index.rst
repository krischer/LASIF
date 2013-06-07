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
opinionated and strict meaning that it forces a certain data and directory
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
these. Just keep in mind to always kepp any data (real waveforms, station
metadata and events) in coordinates that correspond to the physical domain and
all synthetic waveforms in coordinates that correspond to the simulation
domain.

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



Adding Seismic Events
---------------------
All events have to be stored in the *EVENTS* subfolder of the project. They
have to be valid QuakeML files with full moment tensor information. LASIF
provides some convenience methods for this purpose. One can leverage the IRIS
SPUD service (http://www.iris.edu/spud/momenttensor) to get GlobalCMT events.
Simply search for an event and copy the url. The **iris2quakeml** script will
then grab the QuakeML from the url and store an XML file in the current folder.

See :doc:`iris2quakeml` for more information. The LASIF command lines tools
contain a convenience wrapper around it that also makes sure that the event
ends up in the correct folder.


.. code-block:: bash

    $ lasif add_spud_event http://www.iris.edu/spud/momenttensor/959525
    $ lasif add_spud_event http://www.iris.edu/spud/momenttensor/995655

All events can be viewed with

.. code-block:: bash

    $ lasif plot_events


.. plot::

    import lasif.visualization
    map = lasif.visualization.plot_domain(-20, +20, -20, +20, 3.0,
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
    lasif.visualization.plot_events(cat, map)


Waveform Data
-------------
Every inversion needs real data to be able to quantify misfits. The waveform
data for all events are stored in the *DATA* subfolder. The data for each
single event will be stored in a subfolder of the *DATA* folder with the
**same name as the QuakeML file minus the .xml**.

These folder are automatically created and updated each time a lasif command is
executed. The simplest command is

.. code-block:: bash

    $ lasif info

This will result in a directory structure in the fashion of::

    TutorialAnatolia
    |-- CACHE
    |-- DATA
    |   |-- GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
    |   |-- GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9
    |-- EVENTS
    |   |-- GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35.xml
    |   |-- GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9.xml
    |-- LOGS
    |-- MODELS
    |-- OUTPUT
    |-- SOURCE_TIME_FUNCTIONS
    |-- STATIONS
    |   |-- RESP
    |   |-- SEED
    |   |-- StationXML
    |-- SYNTHETICS
    |   |-- GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
    |   |-- GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9
    |-- TEMPLATES
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

    TutorialAnatolia
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
parts of LASIF cannot operate properly.

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
some specification to detail the download. Each event is referred to by its
name which is simply the filename minus the extension. To get a list of all
events in the current project just execute

.. code-block:: bash

    $ lasif list_events

    2 events in project:
        GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
        GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9


To download the waveform data for one event, choose one and run

.. code-block:: bash

    $ lasif download_waveforms GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35


This, dependent on the domain size, event location, and origin time can take a
while. Executing the same command again will only attempt to download data not
already present. All data will be placed in `DATA/EVENT_NAME/raw`.


Downloading Station Data
^^^^^^^^^^^^^^^^^^^^^^^^

LASIF also includes some functionality to download station metadata. It will,
download StationXML and RESP files from IRIS and dataless SEED and RESP files
from ArcLink. It works the same as it does for the waveforms. To download all
stations for one event simply execute

.. code-block:: bash

    $ lasif download_stations GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9

.. note::

    The `lasif download_stations` command will, for the specified event, figure
    what waveform data is present in the `DATA/EVENT_NAME/raw` folder and
    download all missing station metadata information for these files.

At this point, LASIF is able to match available station and waveform
information. To get an overview, of what data is actually stored for the given event, just execute:

.. code-block:: bash

    $ lasif event_info GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9

    Earthquake with 6.1 Mw at AZORES ISLANDS REGION
            Latitude: 37.400, Longitude: -24.380, Depth: 12.0 km
            2007-04-07T07:09:29.500000Z UTC

    Station and waveform information available at 8 stations:

    ===========================================================================
                 id       latitude      longitude      elevation    local depth
    ===========================================================================
            GE.CART        37.5868        -1.0012           65.0            5.0
             GE.MTE        40.3997        -7.5442          815.0            3.0
             GE.SFS        36.4656        -6.2055           21.0            5.0
             IU.PAB        39.5446      -4.349899          950.0            0.0
             PM.MTE        40.3997        -7.5442          815.0            3.0
           PM.PESTR        38.8672        -7.5902          410.0            0.0
            PM.PVAQ        37.4037        -7.7173          200.0            0.0
            WM.CART        37.5868        -1.0012           65.0            5.0


It is furthermore possible to plot the availability information for one event including ray coverage with:

.. code-block:: bash

    $ lasif plot_event GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9


.. plot::

    import lasif.visualization
    map = lasif.visualization.plot_domain(-20, +20, -20, +20, 3.0,
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
    lasif.visualization.plot_events(cat, map)
    ev_lng = -24.38
    ev_lat = 37.4
    stations = {'GE.SFS': {'latitude': 36.4656, 'local_depth': 5.0,
        'elevation': 21.0, 'longitude': -6.2055}, 'PM.MTE': {'latitude':
        40.3997, 'local_depth': 3.0, 'elevation': 815.0, 'longitude': -7.5442},
        'PM.PVAQ': {'latitude': 37.4037, 'local_depth': 0.0, 'elevation':
        200.0, 'longitude': -7.7173}, 'WM.CART': {'latitude': 37.5868,
        'local_depth': 5.0, 'elevation': 65.0, 'longitude': -1.0012}, 'GE.MTE':
        {'latitude': 40.3997, 'local_depth': 3.0, 'elevation': 815.0,
        'longitude': -7.5442}, 'PM.PESTR': {'latitude': 38.8672, 'local_depth':
        0.0, 'elevation': 410.0, 'longitude': -7.5902}, 'GE.CART': {'latitude':
        37.5868, 'local_depth': 5.0, 'elevation': 65.0, 'longitude': -1.0012},
        'IU.PAB': {'latitude': 39.5446, 'local_depth': 0.0, 'elevation': 950.0,
        'longitude': -4.349899}}
    lasif.visualization.plot_stations_for_event(map_object=map,
        station_dict=stations, event_longitude=ev_lng,
        event_latitude=ev_lat)
    # Plot the beachball for one event.
    lasif.visualization.plot_events(cat, map_object=map)


Generating SES3D Input Files
----------------------------

LASIF is currently capable of producing input files for SES3D 4.0. It is very
straightforward and knows what data is available for every event and thus can
generate these files fully automatically.


Preparatory Steps
^^^^^^^^^^^^^^^^^

Before the first input file can be generated some preparatory steps need to be
performed. This is only necessary once at the start or when you make
significant changes to how the simulations are performed.

Input File Templates
********************

At least almost fully automatically. It is necessary to create a template with
the non-derivable configuration values first. This template will then be used
as a basis for all generated input files. It is possible (and encouraged) to
created multiple templates to cover various situations.

To create a basic template (in this case for SES3D 4.0) run:

.. code-block:: bash

    $ lasif generate_input_file_template ses3d_4_0

This will create a (hopefully self-explaining) XML input file template, that **MUST BE EDITED**.

.. code-block:: xml

    <?xml version='1.0' encoding='UTF-8'?>
    <ses3d_4_0_input_file_template>
      <simulation_parameters>
        <number_of_time_steps>500</number_of_time_steps>
        <time_increment>0.75</time_increment>
        <is_dissipative>false</is_dissipative>
      </simulation_parameters>
      <output_directory>../OUTPUT/CHANGE_ME/</output_directory>
      <adjoint_output_parameters>
        <sampling_rate_of_forward_field>10</sampling_rate_of_forward_field>
        <forward_field_output_directory>../OUTPUT/CHANGE_ME/ADJOINT</forward_field_output_directory>
      </adjoint_output_parameters>
      <computational_setup>
        <nx_global>15</nx_global>
        <ny_global>15</ny_global>
        <nz_global>10</nz_global>
        <lagrange_polynomial_degree>4</lagrange_polynomial_degree>
        <px_processors_in_theta_direction>1</px_processors_in_theta_direction>
        <py_processors_in_phi_direction>1</py_processors_in_phi_direction>
        <pz_processors_in_r_direction>1</pz_processors_in_r_direction>
      </computational_setup>
    </ses3d_4_0_input_file_template>

In case something is not fully clear, please refer to the SES3D 4.0 manual or contact the author. It is important to understand that each template file will be used as basis for all generated input files.

Input file templates are again refered to by their filename minus the XML
extension. To get a list of all available templates use:

.. code-block:: bash

    $ lasif list_input_file_templates

    Project has 1 input file template:
            ses3d_4_0_template


You can (and maybe should) rename the actual template files to make it more
descriptive.

Source Time Functions
*********************

The source time function will be dynamically generated for each run. An example
source time function has been generated upon project initialization and is
located in the *SOURCE_TIME_FUNCTIONS* subdirectory.

To create your own source time functions simply copy the already existing one
and modify it. Each source time function has to live in it's own Python file
and a function **source_time_function(npts, delta)** has to be defined in it.
It should return either a list of floats or a numpy array with npts items.

As always, they are referred to via their file name. To get a list of all
available source time functions type:

.. code-block:: bash

    $ lasif list_stf

    Project has 1 defined source time function
            heaviside_60s_500s


It is furthermore possible to get a nice plot for every source time function.
This is useful for visually judging the frequency content that goes into your
simulation. This is done with:

.. code-block:: bash

    $ lasif plot_stf SOURCE_TIME_FUNCTION NPTS DELTA

The number of samples and the sample spacing of any simulation should be known.
SOURCE_TIME_FUNCTION again is the name of the source time function.

.. code-block:: bash

    $ lasif plot_stf heaviside_60s_500s 1500 0.75


.. plot::

    import lasif.visualization
    import obspy
    import numpy as np
    def filtered_heaviside(npts, delta, freqmin, freqmax):
        trace = obspy.Trace(data=np.ones(npts))
        trace.stats.delta = delta
        trace.filter("lowpass", freq=freqmax, corners=5)
        trace.filter("highpass", freq=freqmin, corners=2)
        return trace.data
    data = filtered_heaviside(1500, 0.75, 1.0 / 500.0, 1.0 / 60.0)
    lasif.visualization.plot_tf(data, 0.75)


Input File Generation
^^^^^^^^^^^^^^^^^^^^^

Now that all requirements are fulfilled we can finally generate the input
files. Input files are generated  with the command


.. code-block:: bash

    $ lasif generate_input_files EVENT_NAME TEMPLATE_NAME TYPE SOURCE_TIME_FCT

**TYPE** has to be one of

    * *normal_simulation*
    * *adjoint_forward*
    * *adjoint_reverse*

The other parameters have to correspond to files in the project folder. Please
remember that you can different commands to figure out what files are part of
the project.


.. code-block:: bash

    $ lasif list_events
    2 events in project:
            GCMT_event_AZORES-CAPE_ST._VINCENT_RIDGE_Mag_6.0_2007-2-12-10-35
                    GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9

    $ lasif list_input_file_templates
    Project has 1 input file template:
            ses3d_4_0_template

    $ lasif list_stf
    Project has 1 defined source time function:
            heaviside_60s_500s


Once everything is figured out, actual input files can be generated with:

.. code-block:: bash

    $ lasif generate_input_files GCMT_event_AZORES_ISLANDS_REGION_Mag_6.1_2007-4-7-7-9 \
        ses3d_4_0_template normal_simulation heaviside_60s_500s

    Written files to '.../OUTPUT/input_files___ses3d_4_0_template___2013-03-26T20:04:24.005713'.


If you are working in a rotated domain, all station coordinates and moment
tensors will automatically been rotated accordingly so that the actual
simulation can take place in an unrotated frame of reference.


Together with some models, these file can directly be used to run SES3D. For
the first couple of runs it is likely a good idea to check these file by hand
to verify your setup and potentially also the correctness of this tool suite.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
