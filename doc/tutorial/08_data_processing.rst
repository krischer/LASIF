.. centered:: Last updated on *February 23rd 2018*.


Data Processing
------------------

Data preprocessing is an essential step if one wants to compare data and
seismograms. It serves several purposes:

* Ensure a similar spectral bandwidth between observed and synthetic data to
  enable a meaningful comparison.
* Removing the instrument response and converting the data to the same units
  used for the synthetics (meters).
* Removal of any linear trends and static offset.
* Interpolations to sample observed and synthetic data at exactly the same
  points in time.

The goal of the processing within **LASIF** is to create data that is directly
comparable to simulated data without any more processing.

While the raw unprocessed data are stored in a folder **DATA/EARTHQUAKES**,
the processed data will be stored in a separate directory
**PROCESSED_DATA/EARTHQUAKES/**

Although in principle you can use any processing tool you like, the simplest
option is probably to make use of **LASIF**'s built-in processing. Using it
is trivial: just launch the **process_data** command.

.. code-block:: bash

    $ lasif process_data

or (this is faster as ``-n`` determines the number of processors it will run
on):

.. code-block:: bash

    $ mpirun -n 4 lasif process_data

If the mpirun does not work, we recommend sticking with the non-mpi version
as it is also quite fast.
The mpirun will start a fully parallelized processing run for all data
required for the specified iteration. If you repeat the command, it will only
process data not already processed. An advantage is that you can cancel the
processing at any time and then later on just execute the command again to
continue where you left off.  This usually only needs to be done every couple
of iterations when you decide to go to higher frequencies or add new data.

The processed data will automatically be put in the correct folder and by now
your directory structure should look like this:

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
    │   ├── __pycache__
    │   │   └── preprocessing_function_asdf.cpython-36.pyc
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
    │       ├── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11
    │       │   └── preprocessed_30s_to_50s.h5
    │       └── GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15
    │           └── preprocessed_30s_to_50s.h5
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

.. note::

    You can use any processing tool you want, but you have to adhere to the
    directory structure -- otherwise **LASIF** will not be able to work with
    the data.
    It is also important that the processed filenames are identical to
    the unprocessed ones. And that they are organized into ASDF files.
