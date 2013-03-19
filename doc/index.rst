.. SES3DPy documentation master file, created by
   sphinx-quickstart on Fri Feb  1 15:47:43 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SES3DPy's Documentation!
===================================

SES3DPy is a **data-driven workflow tool** to perform full waveform inversions.
It is opinionated and strict meaning that it forces a certain data and
directory structure. The upside is that it only requires a very minimal amount
of configuration and maintenance. It attempts to gather all necessary
information from the data itself so there is no need to keep index or content
files.

All parts of SES3DPy can work completely on their own. See the class and
function documentation at the end of this document. Furthermore SES3DPy offers
a project based inversion workflow management system which is introduced in the
following tutorial.


Tutorial
========
SES3DPy works with the notion of so called inversion projects. A project is
defined as a series of iterations working on the same physical domain. Where
possible and useful, SES3DPy will use XML files to store information. The
reasoning behind this is twofold. It is easily machine and human readable. It
also serves as a preparatory steps towards a fully database driven full
waveform inversion as all necessary information is already stored in an easily
indexable data format.

Creating a New Project
----------------------
The necessary first step, whether for starting a new inversion or migrating an
already existing inversion to SES3DPy, is to create a new project. In the
following the project will be called **MyInversion**.

.. code-block:: bash

    $ ses3dpy init_project MyInversion

This will create the following directory structure::

    MyInversion
    ├── DATA
    ├── EVENTS
    ├── MODELS
    ├── STATIONS
    │   ├── RESP
    │   ├── SEED
    │   └── StationXML
    ├── SYNTHETICS
    └── simulation_domain.xml


The domain each project works in, is defined in the **simulation_domain.xml**
file. It is a simple, self-explanatory XML format. The nature of SES3D's
coordinate system has the effect that simulation is most efficient in
equatorial regions. Thus it is oftentimes advantageous to rotate the frame of
reference so that the simulation happens close to the equator. SES3DPy first
defines the simulation domain; the actual simulation happens here. Optional
rotation parameters define the physical location of the domain. The coordinate
system for the rotation parameters is described in :py:mod:`ses3dpy.rotations`.
You will have to edit the file to adjust it to your region of interest. It will
look something like the following.

.. code-block:: xml

    <?xml version="1.0" encoding="utf-8"?>
    <domain>
      <name>MyInversion</name>
      <description></description>
      <domain_bounds>
        <minimum_longitude>-20.0</minimum_longitude>
        <maximum_longitude>20.0</maximum_longitude>
        <minimum_latitude>-20.0</minimum_latitude>
        <maximum_latitude>20.0</maximum_latitude>
        <minimum_depth_in_km>0.0</minimum_depth_in_km>
        <maximum_depth_in_km>200.0</maximum_depth_in_km>
      </domain_bounds>
      <domain_rotation>
        <rotation_axis_x>1.0</rotation_axis_x>
        <rotation_axis_y>1.0</rotation_axis_y>
        <rotation_axis_z>1.0</rotation_axis_z>
        <rotation_angle_in_degree>35.0</rotation_angle_in_degree>
      </domain_rotation>
    </domain>

.. note::

    All **ses3dpy** commands work and use the correct project as long as they
    are executed somewhere inside a projects folder structure.

At any point you can have a look at the defined domain with

.. code-block:: bash

    $ cd MyInversion
    $ ses3dpy plot_domain

This will open a window showing the location of the physical domain and the
simulation domain.

.. plot::

    import ses3dpy.visualization
    ses3dpy.visualization.plot_domain(-20, +20, -20, +20, rotation_axis=[1.0,
        1.0, 1.0], rotation_angle_in_degree=35.0, plot_simulation_domain=True)

Adding Seismic Events
---------------------
All events have to be stored in the *EVENTS* subfolder of the project. They
have to be valid QuakeML files with full moment tensor information. SES3DPy
provides some convenience methods for this purpose. One can leverage the IRIS
SPUD service (http://www.iris.edu/spud/momenttensor) to get GlobalCMT events.
Simply search for an event and copy the url. The **iris2quakeml** script will
then grab the QuakeML from the url and store an XML file in the current folder.

See :doc:`iris2quakeml` for more information.

.. code-block:: bash

    $ cd EVENTS
    $ iris2quakeml http://www.iris.edu/spud/momenttensor/878180
    $ iris2quakeml http://www.iris.edu/spud/momenttensor/871125

All events can be viewed with

.. code-block:: bash

    $ ses3dpy plot_events


.. plot::

    import ses3dpy.visualization
    map = ses3dpy.visualization.plot_domain(-20, +20, -20, +20,
        rotation_axis=[1.0, 1.0, 1.0], rotation_angle_in_degree=35.0,
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
    org.latitude = -5.91
    org.longitude = 26.42
    t.m_rr = -2.456e+18
    t.m_tt = 1.035e+18
    t.m_pp = 1.421e+18
    t.m_rt = -1.774e+18
    t.m_rp = -4.48e+17
    t.m_tp = 2.448e+18
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
    org.latitude = -27.92
    org.longitude = 26.88
    t.m_rr = -2.798e+16
    t.m_tt = -1.152e+16
    t.m_pp = 3.949e+16
    t.m_rt = -7e+15
    t.m_rp = 3.66e+15
    t.m_tp = -8.16e+15
    ses3dpy.visualization.plot_events(cat, map)


Waveform Data
-------------
Every inversion needs real data to be able to quantify misfits. The waveform
data for all events are stored in the *DATA* subfolder. The data for each
single event will be stored in a subfolder of the *DATA* folder with the
**same name as the QuakeML file minus the .xml**.

To automatically create the necessary folder for each event run

.. code-block:: bash

    $ ses3dpy update_structure

This will result in a directory structure in the fashion of::

    MyInversion
    ├── DATA
    │   ├── GCMT_event_DEMOCRATIC_REPUBLIC_OF_CONGO_Mag_6.3_1992-9-11-3-57
    │   └── GCMT_event_SOUTH_AFRICA_Mag_5.0_1990-9-26-23-8
    ├── EVENTS
    │   ├── GCMT_event_DEMOCRATIC_REPUBLIC_OF_CONGO_Mag_6.3_1992-9-11-3-57.xml
    │   └── GCMT_event_SOUTH_AFRICA_Mag_5.0_1990-9-26-23-8.xml
    ├── MODELS
    ├── STATIONS
    │   ├── RESP
    │   ├── SEED
    │   └── StationXML
    ├── SYNTHETICS
    │   ├── GCMT_event_DEMOCRATIC_REPUBLIC_OF_CONGO_Mag_6.3_1992-9-11-3-57
    │   └── GCMT_event_SOUTH_AFRICA_Mag_5.0_1990-9-26-23-8
    └── simulation_domain.xml


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
    ├── DATA
        └── GCMT_event_CENTRAL_ITALY_Mag_5.9_2009-4-6-1-32
            ├── raw
                ...
            ├── 100s_to_10s_bandpass
                ...
            └── 200s_to_20s_bandpass
                ...
    ├── SYNTHETICS
        └── GCMT_event_CENTRAL_ITALY_Mag_5.9_2009-4-6-1-32
            ├── inversion_1_100s
                ...
            ├── inversion_2_100s
                ...
            └── inversion_2_50s
                ...
    └── ...

**The user is responsible** for adhering to that structure. Otherwise other
parts of SES3DPy cannot operate properly.

Station Data
------------
SES3DPy needs to know the coordinates and instrument response of each channel.
One way to achieve this to use SAC files, which contain coordinates, and RESP
files containing the response information for each channel. Another possibility
is to use MiniSEED waveform data and the corresponding dataless SEED or
StationXML files.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
