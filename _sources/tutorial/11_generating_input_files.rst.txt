.. centered:: Last updated on *August 12th 2016*.

.. note::

    The following links shows the example project as it should be just before
    step 11. You can use this to check your progress or restart the tutorial at
    this very point.

    `After Step 11: Earth Model Handling <https://github.com/krischer/LASIF_Tutorial/tree/after_step_11_earth_model_handling>`_

Generating Input Files
----------------------

**LASIF** is capable of producing input files for all supported waveform
solvers. It knows what data is available for every event and thus can generate
these files fully automatically.

The iteration files discussed in one of the previous sections are also used to
define the solver settings.

Input File Generation
^^^^^^^^^^^^^^^^^^^^^

The actual input file generation is now very straightforward:


.. code-block:: bash

    $ lasif generate_input_files ITERATION_NAME EVENT_NAME --simulation_type=SIMULATION_TYPE

``SIMULATION_TYPE`` has to be one of

    * ``normal_simulation`` - Use this if you want to get some waveforms, but
      do not need to save the whole forward wavefield.
    * ``adjoint_forward`` - Use this for the forward adjoint simulation. Please
      note that it requires a huge amount of disk space for the forward
      wavefield.
    * ``adjoint_reverse`` - Use this for the adjoint simulation. This requires
      that the adjoint sources have already been calculated. More on that later
      on.

The other parameters should be clear.

For this tutorial you can generate input files for both events with

.. code-block:: bash

    $ lasif generate_input_files 1 GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17 --simulation_type=adjoint_forward
    $ lasif generate_input_files 1 GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20 --simulation_type=adjoint_forward


This will place input files in the ``OUTPUT/input_files`` subdirectory of the
project. In general it is advisable to never delete the input files to
facilitate provenance and reproducibility.

If you are working in a rotated domain, all station coordinates and moment
tensors will automatically be rotated accordingly so that the actual simulation
can take place in an unrotated frame of reference.

Together with the model files for a given iteration model, these files can
directly be used to run SES3D simulations (for other solvers the model part
must be handled seperately). For the first couple of runs it is likely a good
idea to check these files by hand in order to verify your setup.

It is no hassle to run the command twice if you only have two events, for more
events, just run

.. code-block:: bash

    $ lasif generate_all_input_files 1 --simulation_type=adjoint_forward

which generates all input files for any given iteration.
