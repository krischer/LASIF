Model Update
------------

LASIF does not currently aid you with updating the model so this is still a
fairly manual process. We are in the process of correcting this.

Right now it is possible to view the raw gradients from a SES3D simulation.
To do this, simply put them in the a folder according to the following scheme:

``KERNELS/ITERATION_{{ITERATION_NAME}}/{{EVENT_NAME}}``

For the example in the tutorial this results in the two folders:

*  ``KERNELS/ITERATION_1/GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17-14``
*  ``KERNELS/ITERATION_1/GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20-2``

If the folder inside an iteration has the name of an event it is assumed to
be the gradient from that particular event. If it has any other name it can
still be plotted but you have to take care of the meaning.

The ``plot_kernel`` command is used to plot the gradients/kernels and usage
is similar to the ``plot_events`` command.

.. code-block:: bash

    $ lasif plot_kernel GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20-2 1
