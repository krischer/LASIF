.. centered:: Last updated on *February 22th 2018*.

Creating a New Project
----------------------

The necessary first step, whether for starting a new inversion or migrating an
already existing inversion to **LASIF**, is to create a new project with the
:code:`lasif init_project` command. In this  tutorial we will work with a
project called **Tutorial**. The :code:`lasif init_project` command will
create a  new folder in whichever directory the command is executed in.

.. code-block:: bash

    $ lasif init_project Tutorial

    Initialized project in:
        /Users/solvi/PhD/LASIF_Projects/LASIF_Tutorial/Tutorial

This will create the following directory structure. A **LASIF** project is
defined by the files it contains. All information it requires will be
assembled from the available data. In the course of this tutorial you will
learn what piece of data belongs where and how **LASIF** interacts with it.

.. code-block:: none

    Tutorial/
    ├── ADJOINT_SOURCES
    ├── DATA
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    ├── FUNCTIONS
    │   ├── __init__.py
    │   ├── light_preprocessing.py
    │   ├── preprocessing_function_asdf.py
    │   ├── process_data.py
    │   ├── process_synthetics.py
    │   ├── source_time_function.py
    │   └── window_picking_function.py
    ├── GRADIENTS
    ├── MODELS
    ├── OUTPUT
    │   └── LOGS
    ├── PROCESSED_DATA
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    ├── SALVUS_INPUT_FILES
    ├── SETS
    │   ├── WEIGHTS
    │   └── WINDOWS
    ├── SYNTHETICS
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    └── lasif_config.toml


Configuration File
^^^^^^^^^^^^^^^^^^

Each project stores its configuration values in **lasif_config.toml**. This
file should always be located in the root directory of the project as this
file is how LASIF determines the root folder of the project. It uses the
toml format which is a readable file format which should be for the most
part self explanatory. It is important that the names of the inputted values
will not be changed.

Each project stores its configuration values in the **config.xml** file; the
location of this file also determines the root folder of the project. It is
a simple, self-explanatory XML format. Please refer to the comments in the
XML file to infer the meaning of the different settings. Immediately after the
project has been initialized, it will resemble the following:

.. code-block:: none

    # Please fill in this config file before proceeding with using LASIF.

    [lasif_project]
      project_name = "Tutorial"
      description = ""

      # Name of the exodus file used for the simulation. Without a mesh file, LASIF will not work.
      mesh_file = "/Users/solvi/PhD/LASIF_Projects/LASIF_Tutorial/Tutorial/"

      # Number of buffer elements at the domain edges, no events or receivers will be placed there.
      # A minimum amount of 3 is advised.
      num_buffer_elements = 8

      # Type of misift, choose from:
      # [TimeFrequencyPhaseMisfitFichtner2008, L2Norm, CCTimeShift]
      misfit_type = "TimeFrequencyPhaseMisfitFichtner2008"

      [lasif_project.download_settings]
        seconds_before_event = 300.0
        seconds_after_event = 3600.0
        interstation_distance_in_meters = 1000.0
        channel_priorities = [ "BH[Z,N,E]", "LH[Z,N,E]", "HH[Z,N,E]", "EH[Z,N,E]", "MH[Z,N,E]",]
        location_priorities = [ "", "00", "10", "20", "01", "02",]

    # Data processing settings,  high- and lowpass period are given in seconds.
    [data_processing]
      highpass_period = 30.0 # Periods longer than the highpass_period can pass.
      lowpass_period = 50.0 # Periods longer than the lowpass_period will be blocked.
      # Only worry about this if you will reduce the size of the raw data set:
      downsample_period = 1.0 # Minimum period of the period range you will have in your (raw) recordings.

      # You most likely want to keep this setting at true.
      scale_data_to_synthetics = true

    [solver_settings]
        number_of_absorbing_layers = 7
        end_time = 600.0
        time_increment = 0.02
        polynomial_order = 4

        salvus_bin = "salvus_wave/build/salvus"
        number_of_processors = 4
        io_sampling_rate_volume = 20
        io_memory_per_rank_in_MB = 5000
        salvus_call = "mpirun -n 4"

        with_anisotropy = true
        with_attenuation = false

        # Source time function type, currently only "bandpass_filtered_heaviside" is supported
        source_time_function_type = "bandpass_filtered_heaviside"

You will probably have another values in ``end_time`` and ``time_increment``
to follow this tutorial, please modify them to resemble this one.

The **lasif_config.toml** file allows you to tune parameters related to the
processing of your data, download settings, and forward simulation parameters.
This is where you let **LASIF** know where it finds the mesh file to use. Which
source time function and which misfit measurement you want to use. *Currently
only the once specified in this example config file are supported* The mesh
file is how **LASIF** determines the domain used for the simulation. This
mesh needs to be in an exodus file format and we recommend using the
`pymesher <https://gitlab.com/Salvus/salvus_mesher/tree/master>`_ which is
a part of `salvus <http://www.salvus.io>`_. You can plot the domain to make
sure that lasif has read the correct domain for your project.

.. code-block:: bash

    $ lasif plot_domain

This will open a window showing the location of the physical domain and the
simulation domain. The inner contour shows the domain minus the previously
defined boundary width. *Currently it only shows the outer boundary but the
inner boundary will be implemented later*

.. note::

    The map projection and zoom should automatically adjust so that it is suitable
    for the dimensions and location of the chosen domain. If this is not the
    case, please file an issue on the project's Github page.
