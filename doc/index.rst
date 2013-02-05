.. SES3DPy documentation master file, created by
   sphinx-quickstart on Fri Feb  1 15:47:43 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SES3DPy's documentation!
===================================

Contents:

.. toctree::
   :maxdepth: 2

    /rotations
    /windows

All parts of SES3DPy can work completely on their own. See the class and
function documentation at the end of this document. Furthermore SES3DPy offers
a project based inversion workflow management system which is introduced in the
following tutorial.

Tutorial
========
SES3DPy works with the notion of inversion projects. A project is defined as a
series of iterations working on the same physical domain. Where possible and
useful, SES3DPy will use XML files to store information. The reasoning behind
this is twofold. It is easily machine and human readable. It also serves as a
preparatory steps towards a fully database driven full waveform inversion as
all necessary information is already stored in an easily indexable data format.

Creating a new Project
----------------------
The necessary first step, whether for starting a new inversion or migrating an
already existing inversion to SES3DPy, is to create a new project. In the
following the project will be called **MyProject**.

.. code-block:: bash

    $ ses3dpy init_project MyInversion

This will create the following directory structure::

    MyInversion
    |-- DATA
    |-- EVENTS
    |-- MODELS
    |-- SYNTHETICS
    |-- simulation_domain.xml

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
        <minimum_longitude>-10.0</minimum_longitude>
        <maximum_longitude>10.0</maximum_longitude>
        <minimum_latitude>-10.0</minimum_latitude>
        <maximum_latitude>10.0</maximum_latitude>
        <minimum_depth_in_km>0.0</minimum_depth_in_km>
        <maximum_depth_in_km>200.0</maximum_depth_in_km>
      </domain_bounds>
      <domain_rotation>
        <rotation_axis_x>1.0</rotation_axis_x>
        <rotation_axis_y>1.0</rotation_axis_y>
        <rotation_axis_z>1.0</rotation_axis_z>
        <rotation_angle_in_degree>33.0</rotation_angle_in_degree>
      </domain_rotation>
    </domain>

.. note::
    All **ses3dpy** commands work and use the correct project as long as they
    are executed somewhere inside a projects folder structure.

At any point you can have a look at the defined domain with

.. code-block:: bash

    $ ses3dpy show_domain

This will open a window showing the location of the physical domain and the
simulation domain.

.. plot::

    import ses3dpy.visualization
    ses3dpy.visualization.plot_domain(-10, +10, -10, +10, rotation_axis=[1.0,
        1.0, 1.0], rotation_angle_in_degree=33.0)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

