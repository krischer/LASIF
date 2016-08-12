.. centered:: Last updated on *August 12th 2016*.

.. note::

    The following links shows the example project as it should be just before
    step 14. You can use this to check your progress or restart the tutorial at
    this very point.

    `After Step 14: Misfit and Adjoint Sources <https://github.com/krischer/LASIF_Tutorial/tree/after_step_14_misfit_and_adjoint_source>`_

Model Update
------------

**LASIF** per se does not aid you with actually updating the model. This is a
concious design decision as a large part of the research or black art aspect in
full seismic waveform inversions actually comes from the model update,
potential preconditioning, and the actually employed numerical optimization
scheme.

This parts illustrates a simple update. When running a real inversion, please
consider spending a couple of days to scripts model updates and everything else
required around a given LASIF project.


Actual model update
^^^^^^^^^^^^^^^^^^^

Because LASIF does not (yet) aid you with updating the model, you will have
to do this manually. Keep in mind that the kernels SES3D produces initially
are the raw gradients, and unsmoothed. You will probably need to smooth them
before updating the model to get reasonable results without excessive
focusing near sources and/or receivers.


Line Search/Model Update Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At one point you will have to check the total misfit for a certain iteration
and model - be it for a line search or to check a new test model. The
recommended strategy to do this within LASIF is to use "side-iterations", e.g.
in the following we will test the misfit for a model update with a step length
of 0.5:

.. code-block:: bash

    $ lasif create_successive_iteration 1 1_steplength_0.5
    $ lasif migrate_windows 1 1_steplength_0.5
    # Copy synthetics for the test iteration - then calculate both misfits.
    $ mpirun -n 4 lasif compare_misfits 1 1_steplength_0.5


You will notice that the ``compare_misfits`` command always requires two
iterations. This is necessary as any  misfit must be calculated for exactly the
same stations, windows, and weights for two iterations to be comparable.  For
steepest/gradient descent optimization methods it is fine to just take the
current and next iteration, for others like conjugate gradient and L-BFGS make
sure to choose the correct baseline iteration to get comparable misfit
measurements!


Simple Update Script
^^^^^^^^^^^^^^^^^^^^

The following is a short script that demonstrates the principles of how to
perform a model update. It does a simple gradient descent update at various
step lengths that then have to be tested. It requires the initial model as well
as the kernels to be in the SES3D regular grid format and utilizes the Python
tools shipping with SES3D.

.. literalinclude:: make_update.py
   :language: python
