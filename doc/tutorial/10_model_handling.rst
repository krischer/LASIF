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

LASIF has some functionality to view the models. To save a model slice to
file, use the **plot_model** command together with the model name and some
identifying parameters.

.. code-block:: bash

..    $ lasif plot_model True_Model

..    Setup:
..        Latitude: -10.00 - 10.00
..        Longitude: -10.00 - 10.00
..        Depth in km: 0.00 - 471.00
..        Total element count: 10816
..        Total collocation point count: 1352000 (without duplicates: 716625)
..    Memory requirement per component: 2.7 MB
..    Available components: A, B, C, lambda, mu, rhoinv
..    Available derived components: rho, vp, vsh, vsv
..    Parsed components:

..    Enter 'COMPONENT DEPTH' ('quit' to exit):

    $ lasif plot_model True_Model 50 vsv True_Model.050km.vsv.png

This will save a plot directly to file (but will not open a plot on the screen).
.. This will print some information about the model like the available components
.. and the components it can derive from these. Keep in mind that for plotting one
.. of the derived components it potentially has to load two or more components so
.. keep an eye on your machines memory. The tool can currently plot horizontal
.. slices for arbitrary components at arbitrary depths. To do this, simply type
.. the component name and the desired depth in kilometer and hit enter. This opens
.. a new window, e.g. for ``vsv 50``:

.. image:: ../images/vsv_52km.png
    :width: 90%
    :align: center

.. Closing the window will enable you to plot a different component or different
.. depth. To leave the model viewer simply type **quit**.

Another option is to use the model graphical user interface. While still a
bit clunky in operation, it provides a way to explore the models (and once
these have been produced, also kernels for each iteration).

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
