.. centered:: Last updated on *August 12th 2016*.

Customizing LASIF
-----------------

Sometimes the built-in configuration possibilities of LASIF just don't cut
it and you need to change things in a more profound way. To this end LASIF
offers per-project functions that will be called by LASIF on different
occasions.

**If you still feel the need to modify LASIF directly, please contact the
developers.** LASIF aims to be a common tool for full waveform inversions
and if everyone has a custom LASIF version that is not possible. If
necessary we will make it more flexible.

In any case, the per-project functions are stored in the ``FUNCTION``
subfolder of an active LASIF project. If you ever want to update these
function to the latest official version of LASIF just delete the files and
launch any LASIF command.

.. code-block:: none

    FUNCTIONS
    ├── preprocessing_function.py
    ├── process_synthetics.py
    └── window_picking_function.py


Common Features of the Custom Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

    LASIF does not keep track of which function has been called for which
    iteration. This is a choice to make customizing it reasonably easy but it
    does hurt reproducibility and provenance. It is your responsibility to
    store the used functions and make a copy if necessary.
    A simple way to keep track of the differences in these function is to
    branch based on the current iteration.

All of the custom function are passed an
:class:`~lasif.iteration_xml.Iteration` object, thus code like the following
is the preferred way to handle differences in these function per iterations.

.. code-block:: python

    def custom_function(arg1, iteration):
        # Assumes iteration names are numeric. You might need a more
        # complicated setup depending on your project.
        if int(iteration.name) < 10:
            do_this()
        else:
            do_that()


To figure out what these iteration objects can do, launch an interactive lasif
session and play around a bit.

.. code-block:: bash

    $ lasif shell

.. code-block:: python

    In [1]: iteration = comm.iterations.get("1")

    In [2]: iteration.name
    Out[2]: '1'

    In [3]: iteration.get_process_params()
    Out[3]:
    {'dt': 0.3,
     'highpass': 0.01,
     'lowpass': 0.025,
     'npts': 2000,
     'stf': 'Filtered Heaviside'}


Custom Preprocessing
^^^^^^^^^^^^^^^^^^^^

The preprocessing function is used to preprocess all the data. It has the
following function signature:

.. code-block:: python

    def preprocessing_function(processing_info, iteration):
        ...

It is supposed to read the ``procesing_info["input_filename"]`` file, apply
the defined processing, and write it to ``processing_info["output_filename"]``.
It does not have to return anything; raise an error if something does not
work as expected.

The ``processing_info`` object also contains event and station information.

.. code-block:: python

    {'event_information': {
        'depth_in_km': 22.0,
        'event_name': 'GCMT_event_VANCOUVER_ISLAND...',
        'filename': '/.../GCMT_event_VANCOUVER_ISLAND....xml',
        'latitude': 49.53,
        'longitude': -126.89,
        'm_pp': 2.22e+18,
        'm_rp': -2.78e+18,
        'm_rr': -6.15e+17,
        'm_rt': 1.98e+17,
        'm_tp': 5.14e+18,
        'm_tt': -1.61e+18,
        'magnitude': 6.5,
        'magnitude_type': 'Mwc',
        'origin_time': UTCDateTime(2011, 9, 9, 19, 41, 34, 200000),
        'region': u'VANCOUVER ISLAND, CANADA REGION'},
     'input_filename': u'/.../raw/7D.FN01A..HHZ.mseed',
     'output_filename': u'/.../processed_.../7D.FN01A..HHZ.mseed',
     'process_params': {
        'dt': 0.75,
        'highpass': 0.007142857142857143,
        'lowpass': 0.0125,
        'npts': 2000,
        'stf': 'Filtered Heaviside'},
     'station_coordinates': {
        'elevation_in_m': -54.0,
        'latitude': 46.882,
        'local_depth_in_m': None,
        'longitude': -124.3337},
     'station_filename': u'/.../STATIONS/RESP/RESP.7D.FN01A..HH*'}



Custom Synthetic Data Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LASIF, by default, does not apply any processing (except rotations and
component flips to get them to ZNE) to the synthetics but just
uses them as is. If your workflow requires the synthetics to be processed,
please do it in this function. It will be applied on the fly to each
up-to-three component :class:`~obspy.core.stream.Stream` object of synthetic
data.

The default implementation is this:

.. code-block:: python

    def process_synthetics(st, iteration, event):
        return st

This is very useful for processing the synthetics in any fashion or to shift
them time and similar endeavours. Make sure it returns a
:class:`~obspy.core.stream.Stream` object.

``iteration`` and ``event`` are the :class:`~lasif.iteration_xml.Iteration`
object of the current iteration and a dictionary containing information about
the data's event, respectively.


Customize Window Picking
^^^^^^^^^^^^^^^^^^^^^^^^

Use this function to customize the window picking of LASIF, or even use a
completely different window picking algorithm. Its function signature is:


.. code-block:: python

    def window_picking_function(data_trace, synthetic_trace, event_latitude,
                                event_longitude, event_depth_in_km,
                                station_latitude, station_longitude,
                                minimum_period, maximum_period,
                                iteration):
        ...

        # Make sure it return a list of tuples, each denoting start and
        # endtime for each picked window.
        return [(obspy.UTCDateTime(...), obspy.UTCDateTime(...)),
                (obspy.UTCDateTime(...), obspy.UTCDateTime(...))]


Customize the Source Time Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This function is used to generate source time functions for SES3D. If you
don't use SES3D for the numerical wavefield simulations you can ignore this.
Its function signature is:


.. code-block:: python


    def source_time_function(npts, delta, freqmin, freqmax, iteration):
        ...
        # Make sure it returns a float64 NumPy array with `npts` samples.
        return np.array(data, dtype=np.float64)
