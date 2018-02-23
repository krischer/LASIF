.. centered:: Last updated on *February 23rd 2018*.


Synthetics
----------

Now that everything has been set up, you have to actually perform the
simulations.  Please keep in mind that the adjoint forward simulations require
a very large amount of disk space due to the need to store the forward
wavefield. If you only want to make some waveforms you can remove the
``--save-wavefield-file wavefield.h5`` and ``--save-fields adjoint`` flags
from your ``run_salvus.sh`` files. If you want to keep using the tutorial but
you do not want to calculate synthetics yourself. You can copy synthetics
from your **LASIF** directory.

.. code-block:: bash

    # Position yourself in the root directory of the project.
    $ cp -r {lasif folder}/lasif/tests/data/ExampleProject/SYNTHETICS/EARTHQUAKES/ITERATION_1/* ./SYNTHETICS/EARTHQUAKES/ITERATION_1/

This should copy some synthetics to the correct folder in your tutorial
project. Your directory structure should look similar to this:

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
    │   │   ├── preprocessing_function_asdf.cpython-36.pyc
    │   │   └── source_time_function.cpython-36.pyc
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
    │       ├── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11
    │       │   └── forward
    │       │       ├── Heaviside.h5
    │       │       ├── receivers.toml
    │       │       ├── receivers_paraview.csv
    │       │       ├── run_salvus.sh
    │       │       ├── source.toml
    │       │       └── source_paraview.csv
    │       └── GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15
    │           └── forward
    │               ├── Heaviside.h5
    │               ├── receivers.toml
    │               ├── receivers_paraview.csv
    │               ├── run_salvus.sh
    │               ├── source.toml
    │               └── source_paraview.csv
    ├── SETS
    │   ├── WEIGHTS
    │   │   └── WEIGHTS_A
    │   │       └── WEIGHTS_A.toml
    │   └── WINDOWS
    ├── SYNTHETICS
    │   ├── CORRELATIONS
    │   └── EARTHQUAKES
    │       └── ITERATION_1
    │           ├── GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11
    │           │   ├── receivers.h5
    │           │   ├── stderr
    │           │   └── stdout
    │           └── GCMT_event_TURKEY_Mag_5.9_2011-5-19-20-15
    │               ├── receivers.h5
    │               ├── stderr
    │               └── stdout
    └── lasif_config.toml

Now we can look at the misfit between data and synthetics and later compute the
misfits and adjoint sources.
