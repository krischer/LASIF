Installation
============

**LASIF** is written for Python 2.7.x, support for Python 3 will follow soon. A
working Python installation with the most common scientific libraries installed
will be assumed for the following instructions.


Dependencies
------------

**LASIF** depends on a number of third-party libraries.

* `ObsPy <http://www.obspy.org/>`_ (in a recent repository version - the version will be fixed at a certain point in the future)
* The `matplotlib basemap toolkit <http://matplotlib.org/basemap/>`_
* `wfs_input_generator <http://github.com/krischer/wfs_input_generator>`_
* requests
* progressbar
* geographiclib
* colorama
* joblib

The later five should prove no trouble installing via pip. They will also be
automatically installed when running **LASIF**'s setup script.

.. code-block:: bash

    $ pip install requests progressbar geographiclib colorama joblib

ObsPy and basemap are slightly more complicated to install. Please refer to the
projects websites for more detailed instructions.

Both, the *wfs_input_generator* module and **LASIF** are in active development
and thus should be installed as a develop installation so they can be easily
upgraded via git:

.. code-block:: bash

    $ git clone https://github.com/krischer/wfs_input_generator.git
    $ cd wfs_input_generator
    $ pip install -v -e .



Installing LASIF
----------------

The actual **LASIF** module can then be installed with

.. code-block:: bash

    $ git clone https://github.com/krischer/LASIF.git
    $ cd LASIF
    $ pip install -v -e .


After the installation one should run the tests to ensure everything is
installed correctly and works as intended on your machine.
