Generating SES3D Input Files
----------------------------

LASIF is currently capable of producing input files for SES3D 4.1. It is very
straightforward and knows what data is available for every event and thus can
generate these files fully automatically. In the future it might be worth
investigating automatic job submission to high performance machines as this is
essentially just repetitive and error-prone work.

The iteration files discussed in one of the previous sections are also used
to define the solver settings.

Input File Generation
^^^^^^^^^^^^^^^^^^^^^

The actual input file generation is now very straightforward:


.. code-block:: bash

    $ lasif generate_input_files ITERATION_NAME EVENT_NAME --simulation_type=SIMULATION_TYPE

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

    $ lasif generate_input_files 1 GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17 --simulation_type=adjoint_forward
    $ lasif generate_input_files 1 GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20 --simulation_type=adjoint_forward


This will place input files in the *OUTPUT* subdirectory of the project. In
general it is advisable to never delete the input files to facilitate
provenance and reproducibility.

If you are working in a rotated domain, all station coordinates and moment
tensors will automatically be rotated accordingly so that the actual simulation
can take place in an unrotated frame of reference.

Together with a models, these files can directly be used to run SES3D. For
the first couple of runs it is likely a good idea to check these file by hand
to verify your setup.
