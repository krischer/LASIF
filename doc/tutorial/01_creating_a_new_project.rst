Creating a New Project
----------------------

The necessary first step, whether for starting a new inversion or migrating an
already existing inversion to LASIF, is to create a new project with the
:code:`lasif init_project` command. In this  tutorial we will work with a
project called **Tutorial**. The :code:`lasif init_project` command will
create a  new folder in whichever directory the command is executed in.

.. code-block:: bash

    $ lasif init_project Tutorial

    Initialized project in:
        /Users/lion/workspace/temp/LASIF_Tutorial/Tutorial

This will create the following directory structure. A **LASIF** project is
defined by the files it contains. All information it requires will be
assembled from the available data. In the course of this tutorial you will
learn what piece of data belongs where.

.. code-block:: none

    Tutorial
    ├── ADJOINT_SOURCES_AND_WINDOWS
    │   ├── ADJOINT_SOURCES
    │   └── WINDOWS
    ├── CACHE
    ├── DATA
    ├── EVENTS
    ├── ITERATIONS
    ├── KERNELS
    ├── LOGS
    ├── MODELS
    ├── OUTPUT
    ├── STATIONS
    │   ├── RESP
    │   ├── SEED
    │   └── StationXML
    ├── SYNTHETICS
    ├── WAVEFIELDS
    └── config.xml


Each project stores its configuration values in the **config.xml** file; the
location of this file also determines the root folder of the project. It is
a simple, self-explanatory XML format. Immediately after the project has been
initialized it looks akin to the following:

.. code-block:: xml

    <?xml version='1.0' encoding='UTF-8'?>
    <lasif_project>
      <name>Tutorial</name>
      <description></description>
      <download_settings>
        <arclink_username></arclink_username>
        <seconds_before_event>300</seconds_before_event>
        <seconds_after_event>3600</seconds_after_event>
      </download_settings>
      <domain>
        <domain_bounds>
          <minimum_longitude>-20</minimum_longitude>
          <maximum_longitude>20</maximum_longitude>
          <minimum_latitude>-20</minimum_latitude>
          <maximum_latitude>20</maximum_latitude>
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


* ``name`` is the project's name.
* ``description`` can be any further useful information about the project. This
  is not used by LASIF but potentially useful for documentation purposes.
* The ``arclink_username`` tag should be your email. It will be sent with all
  requests to the ArcLink network. They ask for it in case they have to contact
  you for whatever reason. Please provide a real email address. Must not be
  empty.
* ``seconds_before_event``: Used by the waveform download scripts. It will
  attempt to download this many seconds for every waveform before the origin of
  the associated event.
* ``seconds_after_event``: Used by the waveform download scripts. It will
  attempt to download this many seconds for every waveform after the origin of
  the associated event. Adapt this to the size of your inversion domain.
* The ``domain`` settings will be explained in more detail in the following
  paragraphs.
* The ``boundary_width_in_degree`` tag is use to be able to take care of the
  boundary conditions, e.g. no data will be downloaded within
  ``boundary_width_in_degree`` distance to the domain border.

The nature of SES3D's coordinate system has the effect that simulation is most
efficient in equatorial regions. Furthermore a domain can only be  specified by
minimum and maximum extends as it works with spherical sections. Thus it is
oftentimes  advantageous to rotate the frame of reference so that the
simulation happens close to the equator. LASIF first defines the simulation
domain; the actual simulation happens there. Optional rotation parameters
define the physical location of the domain. The coordinate system for the
rotation parameters is described in :py:mod:`lasif.rotations`.  You will have
to edit the ``config.xml`` file to adjust it to your region of interest.

LASIF handles all rotations necessary so the user never needs to worry about
these. Just keep in mind to always keep any data (real waveforms, station
metadata and events) in coordinates that correspond to the physical domain and
all synthetic waveforms in coordinates that correspond to the simulation
domain.

For this tutorial we are going to work in a rotated domain across Europe.
Please change the ``config.xml`` file to reflect the following domain
settings.

* Latitude: ``-10.0° - 10.0°``
* Longitude: ``-10.0° - 10.0°``
* Depth: ``0 km - 471 km``
* Boundary width in degree: ``2.5°``
* Rotation axis: ``1.0, 1.0, 0.2``
* Rotation angle: ``-65.0°``

In general one should only work with data not affected by the boundary
conditions. SES3D utilizes perfectly matched layers boundary conditions (PML).
It is not advisable to use data that traverses these layers. SES3D defaults
to two layer but more are possible. For this tutorial we will only consider
data which is at least three elements away from the border in a an attempt
to avoid unphysical influences of the boundary conditions. This amounts to
``2.5°``.

At any point you can have a look at the defined domain with

.. code-block:: bash

    $ lasif plot_domain

This will open a window showing the location of the physical domain and the
simulation domain. The inner contour shows the domain minus the previously
defined boundary width.

.. plot::

    import lasif.visualization
    lasif.visualization.plot_domain(-10.0, 10.0, -10.0, 10.0, 2.5,
        rotation_axis=[1.0, 1.0, 0.2], rotation_angle_in_degree=-65.0,
        plot_simulation_domain=True, zoom=True)

.. note::

    The map projection and zoom should automatically adjust so it is suitable
    for the dimensions and location of the chosen domain. If that is not the
    case please file an issue on the project's Github page.

Assuming you carefully followed this part the ``config.xml`` file should
look like this.

.. code-block:: xml

    <?xml version='1.0' encoding='UTF-8'?>
    <lasif_project>
      <name>Tutorial</name>
      <description></description>
      <download_settings>
        <arclink_username></arclink_username>
        <seconds_before_event>300</seconds_before_event>
        <seconds_after_event>3600</seconds_after_event>
      </download_settings>
      <domain>
        <domain_bounds>
          <minimum_longitude>-10</minimum_longitude>
          <maximum_longitude>10</maximum_longitude>
          <minimum_latitude>-10</minimum_latitude>
          <maximum_latitude>10</maximum_latitude>
          <minimum_depth_in_km>0.0</minimum_depth_in_km>
          <maximum_depth_in_km>471.0</maximum_depth_in_km>
          <boundary_width_in_degree>2.5</boundary_width_in_degree>
        </domain_bounds>
        <domain_rotation>
          <rotation_axis_x>1.0</rotation_axis_x>
          <rotation_axis_y>1.0</rotation_axis_y>
          <rotation_axis_z>0.2</rotation_axis_z>
          <rotation_angle_in_degree>-65.0</rotation_angle_in_degree>
        </domain_rotation>
      </domain>
    </lasif_project>


.. note::

    The true synthetic model is PREM with a small positive Gaussian anomaly  in
    the center (latitude=longitude=0) at a depth of 70 km applied to the  P and
    both S-wave velocities. The amplitude of the anomaly is 0.3 km/s with
    the sigma being 200 km in horizontal and 50 km in the vertical direction.


