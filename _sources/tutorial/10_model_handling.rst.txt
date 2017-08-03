.. centered:: Last updated on *August 12th 2016*.

Earth Model Handling
--------------------

**LASIF** has a very simple approach to handling Earth Models. The idea is to
have one Earth model per iteration with ideally the same name as the
iteration. **LASIF** does not yet perform any work with the Earth models,
it merely offers an organized way of storing and visualizing them.

**LASIF** can directly deal with the models used in SES3D (that is, the
ready-to-go models with which a simulation is run). Each model has to
be placed in a subfolder of ``MODELS``. The folder name will again be used to
identify the model.

For the tutorial, we will work with two models: one based on PREM which we
will use as the initial model and one with a perturbation which will be
considered the true model for the tutorial. Download both models
:download:`here <../downloads/models.tar.bz2>` and place them in the
``MODELS`` folder.


.. note::

    The true synthetic model is PREM with a small positive Gaussian anomaly  in
    the center (latitude=longitude=0) at a depth of 70 km applied to the  P and
    both S-wave velocities. The amplitude of the anomaly is 0.3 km/s with
    the sigma being 200 km in horizontal and 50 km in the vertical direction.



Now you are able to use the **list_models** commands.

.. code-block:: bash

    $ lasif list_models

    2 model in project:
        Iteration_1
        True_Model

LASIF has some functionality to view the models. To save a model slice to file,
use the ``plot_model`` command together with the model name and some
identifying parameters.

.. code-block:: bash

    $ lasif plot_model True_Model 50 vsv -

This will pop open a window. If you want to save the image to a file, just
replace ``-`` with a filename.

.. image:: ../images/vsv_52km.png
    :width: 90%
    :align: center

Another option is to use

.. code-block:: bash

    $ lasif launch_model_gui

This opens a more interactive viewer in which the user can choose which
model to look at, walk through the different depth levels, and explore
different (derived) components:

.. image:: ../images/model_gui.screenshot.2016-06-14.png
    :width: 90%
    :align: center

When running the numerical simulations, **the user is responsible to choose and
copy the correct earth model file**.
