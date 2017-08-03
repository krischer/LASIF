.. centered:: Last updated on *August 12th 2016*.

Next Iterations
---------------

Once you updated your model it is time to go to the next iteration.
Depending on the state of the inversion and your goals you might which to
add more events and data. Just use the tools introduced in the previous
sections. Please never delete data you used in any of the iterations,
otherwise you hurt the reproducibility of your inversion.

You essentially have two options to go from one iteration to the next:
Create a new iteration or update from the previous one. In any case,
you are responsible for the name of the iteration.

Create a new iteration from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we will create an iteration ``2`` for the tutorial with the same settings
as iteration ``1``.

.. code-block:: bash

    $ lasif create_new_iteration 2 40.0 100.0 SES3D_4_1

Using this command will create a new iteration with all the data currently in
the project, thus all events and all waveforms it can find. This is also
useful if you want to choose a different waveform solver or a different
frequency range for the next iteration.


Deriving from an existing iteration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The other option is to derive a new iteration from an existing iteration. In
this case, the solver and preprocessing settings will remain the same,
as will the used data. Any data not used for the previous iteration will
also not be used for this iteration.

.. code-block:: bash

    $ lasif create_successive_iteration 1 2

This commando takes an existing iteration ``1`` and uses it as a template
for iteration ``2``.


Migrating the misfit windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once the windows for the misfit calculation have been determined for an
iteration they can usually be used for a couple of iterations before they have
to be determined again. The following command will take all windows defined
in iteration ``1`` and transfer them to iteration ``2``. The synthetics for
iteration ``2`` already need to be available. Depending on the size of the
iteration, this might take a while as it also calculates the adjoint sources
for each window.

.. code-block:: bash

    $ lasif migrate_windows 1 2

Please keep in mind that this command will not create windows for data not
present in the first iteration.

The ``finalize_adjoint_sources`` command can then be used generate the
adjoint sources in a format suitable for SES3D to finish the iteration.
