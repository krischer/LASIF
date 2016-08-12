.. centered:: Last updated on *August 12th 2016*.

.. note::

    The following links shows the example project as it should be just before
    step 9. You can use this to check your progress or restart the tutorial at
    this very point.

    `After Step 9: Defining a new Iteration <https://github.com/krischer/LASIF_Tutorial/tree/after_step_9_defining_a_new_iteration>`_

Data Preprocessing
------------------

.. note::

    You do not actually need to do this for the tutorial as the proprocessed
    data has already been downloaded but this will be required for real
    inversions.

Data preprocessing is an essential step if one wants to compare data and
seismograms. It serves several purposes:

* Ensure a similar spectral bandwidth between observed and synthetic data to
  enable a meaningful comparison.
* Removing the instrument response and converting the data to the same units
  used for the synthetics (usually ``m/s``).
* Removal of any linear trends and static offset.
* Interpolations to sample observed and synthetic data at exactly the same
  points in time.

The goal of the preprocessing within **LASIF** is to create data that is directly
comparable to simulated data without any more processing.

While the raw unprocessed data are stored in a folder ``{{EVENT}}/raw``, the
preprocessed data will be stored in a separate directory within each event,
identified via the name (values are valid for the tutorial)::

    preprocessed_hp_0.01000_lp_0.02500_npts_2000_dt_0.300000

Or in Python terms:

.. code-block:: python

    highpass = 1.0 / 100.0
    lowpass = 1.0 / 40.0
    npts = 2000
    dt = 0.3

    processing_tag = ("preprocessed_hp_{highpass:.5f}_lp_{lowpass:.5f}_"
                      "npts_{npts}_dt_{dt:5f}").format(highpass=highpass, lowpass=lowpass,
                                                       npts=npts, dt=dt)

If you feel that additional identifiers are needed to uniquely identify the
applied processing (in the limited setting of being useful for the here
performed waveform inversion) please contact the **LASIF** developers.

Although in principle you can use any processing tool you like, the simplest
option is probably to make use of **LASIF**'s built-in preprocessing. Using it
is trivial: just launch the **preprocess_data** command together with the
iteration name.

.. code-block:: bash

    $ lasif preprocess_data 1

or (this is faster as ``-n`` determines the number of processors it will run
on):


.. code-block:: bash

    $ mpirun -n 4 lasif preprocess_data 1


This will start a fully parallelized preprocessing run for all data required
for the specified iteration. If you repeat the command, it will only process
data not already processed. An advantage is that you can cancel the processing
at any time and then later on just execute the command again to continue where
you left off.  This usually only needs to be done every couple of iterations
when you decide to go to higher frequencies or add new data.

The preprocessed data will automatically be put in the correct folder.

.. note::

    You can use any processing tool you want, but you have to adhere to the
    directory structure -- otherwise **LASIF** will not be able to work with
    the data.
    It is also important that the processed filenames are identical to
    the unprocessed ones.
