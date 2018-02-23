.. centered:: Last updated on *February 23rd 2018*.

Defining a New Iteration
------------------------

In **LASIF** you can organize your synthetics and velocity models through
different iterations. To create a new iteration you execute:

.. code-block:: bash

    $ lasif set_up_iteration 1

Same command can also be used to remove iteration. The command creates a few
directories which will be used later to organize synthetics, adjoint sources
and velocity models. By now your directory structure should look similar to
this:

.. code-block:: none

    Tutorial
    ├── ADJOINT_SOURCES
    │   └── ITERATION_1
    ├── DATA
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    │       ├── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11.h5
    │       └── GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15.h5
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
    │   ├── ITERATION_1
    │   └── Turkey.e
    ├── OUTPUT
    │   ├── LOGS
    │   └── raydensity_plots
    │       └── 2018-02-23T08-38-48__raydensity
    │           └── raydensity.png
    ├── PROCESSED_DATA
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    ├── SALVUS_INPUT_FILES
    │   └── ITERATION_1
    ├── SETS
    │   ├── WEIGHTS
    │   │   └── WEIGHTS_A
    │   │       └── WEIGHTS_A.toml
    │   └── WINDOWS
    ├── SYNTHETICS
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    │       └── ITERATION_1
    └── lasif_config.toml

The Turkey.e file is the mesh that we use in the project. The organisation of
velocity models into iterations is manual and optional but **LASIF** anyway
creates the directory structure for it if you want to use it. To get a list
of iterations, use:

.. code-block:: bash

    $ lasif list_iterations

    There is 1 in this project
    Iteration known to LASIF:

    ITERATION_1

.. note::

    As mentioned before, it is entirely possible to add new events at a later
    stage during an inversion.

You will always have to set up an iteration before you can use the misfit GUI.