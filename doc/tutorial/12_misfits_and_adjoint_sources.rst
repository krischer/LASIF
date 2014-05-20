Misfit and Adjoint Source Calculation
-------------------------------------

In order to simulate the adjoint wavefield one needs to calculate the adjoint
sources. An adjoint source is usually dependent on the misfit between the
synthetics and real data.

LASIF currently supports misfits in the time-frequency domain as defined by
Fichtner, 2008. Great care has to be taken to avoid cycle skips/phase jumps
between synthetics and data. This is achieved by careful windowing.

Weighting Scheme
^^^^^^^^^^^^^^^^

You might have noticed that at various points it has been possible to
distribute weights. These weights all contribute to the final adjoint source.
The inversion scheme requires one adjoint source per iteration, event, station,
and component.

In each iteration's XML file you can specify the weight for each event, denoted
as :math:`w_{event}` in the following. The weight is allowed to range between
0.0 and 1.0. A weight of 0.0 corresponds to no weight at all, e.g. skip the
event and a weight of 1.0 is the maximum weight.

Within each event is possible to weight each station separately, in the
following named :math:`w_{station}`. The station weights can also range from
0.0 to 1.0 and follow the same logic as the event weights.

You can furthermore choose an arbitrary number of windows per component for
which the misfit and adjoint source will be calculated. Each window has a
separate weight with the only limitation being that the weight has to be
positive. Assuming :math:`N` windows in a given component, the corresponding
adjoint sources will be called :math:`ad\_src_{1..N}` while their weights are
:math:`w_{1..N}`.

The final adjoint source for every component will be calculated according to
the following formula:

.. math::

   adj\_source = w_{event} \cdot w_{station} \cdot \frac{1}{\sum_{i=1}^N w_i} \cdot \sum_{i=1}^N w_i \cdot ad\_src_i

Misfit GUI
^^^^^^^^^^

LASIF comes with a graphical utility dubbed the Misfit GUI, that helps to pick
correct windows.

To launch it, execute the **launch_misfit_gui** together with the iteration
name and the event name.

.. code-block:: bash

    $ lasif launch_misfit_gui 1 GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11


This will open a window akin to the following:

.. image:: ../images/misfit_gui.png
    :width: 90%
    :align: center

It is essentially partitioned into three parts. The top part is devoted to plot
all three traces of a single station. The bottom left part shows a
representation of the misfit for the last chosen window and the bottom left
part shows a simple map.

.. note::

    The current interface is based purely on matplotlib. This has the advantage
    of keeping dependencies to minimum. Unfortunately matplotlib is not a GUI
    toolkit and therefore the GUI is not particularly pleasing from a UI/UX
    point of view. Some operations might feel clunky. We might move to a proper
    GUI toolkit in the future.

With the **Next** and **Prev** button you can jump from one station to the
next. The **Reset Station** button will remove all windows for the current
station.

The weight for any window has to be chosen before the windows are picked. To
chose the current weight, press the **w** key. At this point, the weight box
will be red. Now simply type the desired new weight and press **Enter** to
finish setting the new weight. All windows chosen from this point on will
be assigned this weight.

To actually choose a window simply drag in any of the waveform windows. Upon
mouse button release the window will be saved and the adjoint source will be
calculated. The number in the top left of each chosen window reflects the
weight for that window.

Right clicking on an already existing window will delete it, left clicking will
plot the misfit once again.

.. note::

    At any point you can press **h** to get an up-to-date help text for the
    GUI.


Final Adjoint Source Calculation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During window selection the adjoint source for each chosen window will be
stored separately. To combine them to, apply the weighting scheme and convert
them to a format, that SES3D can actually use, run the
**finalize_adjoint_sources** command with the iteration name and the event
name.

.. code-block:: bash

    $ lasif finalize_adjoint_sources 1 GCMT_event_TURKEY_Mag_5.1_2010-3-24-14-11

This will also rotate the adjoint sources to the frame of reference used in the
simulations.

If you pick any more windows or change them in any way, you need to run the
command again.
