Installation, Testing & DevInfo
===============================

Installation
------------

**LASIF** is written for Python 2.7.x, support for Python 3 will follow soon. A
working Python installation with the most common scientific libraries installed
will be assumed for the following instructions.


Dependencies
^^^^^^^^^^^^

**LASIF** depends on a number of third-party libraries.

* `ObsPy <http://www.obspy.org/>`_ (in a recent repository version - the version will be fixed at a certain point in the future)
* The `matplotlib basemap toolkit <http://matplotlib.org/basemap/>`_
* `wfs_input_generator <http://github.com/krischer/wfs_input_generator>`_
* progressbar
* geographiclib
* colorama
* joblib

For the webinterface, you additionally need

* Flask
* Flask-Cache
* geojson

And for running the tests you need

* pytest
* mock
* nose
* flake8


ObsPy and basemap might prove complicated to install. Please refer to the
projects` websites for more detailed instructions.

The *wfs_input_generator* module currently has to be checked out via git:

.. code-block:: bash

    $ git clone https://github.com/krischer/wfs_input_generator.git
    $ cd wfs_input_generator
    $ pip install -v -e .

Make sure **ObsPy**, **basemap**, and the **wfs_input_generator** are
installed before proceeding.


Installing LASIF
^^^^^^^^^^^^^^^^

The actual **LASIF** module can then be installed with

.. code-block:: bash

    $ git clone https://github.com/krischer/LASIF.git
    $ cd LASIF
    $ pip install -v -e .

This will also install all further dependencies.

After the installation one should run the tests to ensure everything is
installed correctly and works as intended on your machine.


Parallel Processing in LASIF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LASIF is parallel in parts to speed up some computations. This unfortunately
is troublesome with Python and NumPy due to incompatbilities with the
employed linear algebra packages and forked processes. LASIF should be smart
enough to protect you from any failures but this potentially means that it
will only be able to use a single processor.

If you experience this, read these `notes on parallel processing with Python
and ObsPy
<https://github.com/obspy/obspy/wiki/Notes-on-Parallel-Processing
-with-Python-and-ObsPy>`_ for some more details and information on how to
fix it.



Testing
-------

Requirements and Rational
^^^^^^^^^^^^^^^^^^^^^^^^^

**LASIF** evolved into a fairly complex piece of code and extensive testing is
required to assure that it works as expected.

To run the tests, you need to install four additional modules:

.. code-block:: bash

    $ pip install pytest mock nose flake8

The `pytest <http://pytest.org>`_ module is an alternative testing framework
for Python offering powerful test discovery features and a no-boilerplate
approach to syntax. It furthermore provides a nice functional approach to
writing tests for complex environments facilitating proper tests for **LASIF**.

Many operations in **LASIF** are computationally expensive or have side effects
like requiring online access making them not particularly well suited to being
testing. In order to verify some of the complex interactions within **LASIF**
dummy objects are used. These are provided by the
`mock <http://www.voidspace.org.uk/python/mock/>`_ package.

**LASIF** contains some graphical functionality outputting maps and other
plots.  matplotlib's testing facilities are reused in **LASIF** which in turn
require the `nose <http://nose.readthedocs.org/en/latest/>`_ testing framework
to be installed.

The `flake8 <http://flake8.readthedocs.org/en/2.0/>`_ package is used to make
sure that **LASIF**'s code base adhere to the
`PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ standard.

Running the Tests
^^^^^^^^^^^^^^^^^

To run the tests, cd somewhere into the **LASIF** code base and type


.. code-block:: bash

    $ py.test

This will recursively find and execute all tests below the current working
directory.

The py.test command accepts a large number of additional parameters, e.g.

.. code-block:: bash

    # Execute only tests within test_project.py.
    $ py.test test_project.py

    # Print stdout and stderr and do not capture it.
    $ py.test -s

    # Execute only tests whose name contains the string 'some_string'.
    $ py.test -k some_string


If your machine has multiple cores, the processing can also be sped up
quite a bit by using `pytest-xdist` which can be installed via pip.

.. code-block:: bash

    $ pip install pytest-xdist

It enables to distribute the test across cores. To run on, for example, eight
cores, use

.. code-block:: bash

    $ py.test -n 8


For more information please read the
`pytest documentation <http://pytest.org/>`_.




Developer Information
---------------------


The following rules should be followed when developing for **LASIF**:

* **LASIF** is written entirely in Python.
* C/Fortran code with proper bindings can be used to improve performance where
  necessary. Cython is also an accepted alternative.
* `Document <http://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/>`_ the
  code.
* Adhere to `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_.
* All contributed code must be contributed under the GPLv3.
* Write tests where reasonable.

    * **LASIF** utilizes `Travis CI <https://travis-ci.org/krischer/LASIF>`_
      for continuous integration testing. This means that every commit will be
      automatically tested and the responsible developer will receive an email
      in case her/his commit breaks **LASIF**.
    * The tests also verify the PEP8 conformance of the entire code base.


Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

`sphinx` is used to build the documentation so it needs to be installed. The
theme is the standalone readthedocs theme. We will use the most up-to-date
repository version here.

.. code-block:: bash

    $ pip install sphinx
    $ pip install https://github.com/snide/sphinx_rtd_theme/archive/master.zip

To actually build the documentation (in this case in the HTML format), run

.. code-block:: bash

    $ cd doc
    $ make html

This might take a while if run for the first time. Subsequent runs are faster.


Terminology
^^^^^^^^^^^

In order to ease development, a consistent terminology should be used
throughout the code base.

Assume a channel with a SEED identifier being equal to `NET.STA.LOC.CHA`, then
the separate components should be called:

* **channel_id**: `NET.STA.LOC.CHA`
* **station_id**: `NET.STA`
* **network_code** or **network**: `NET`
* **station_code** or **station**: `STA`
* **location_code** or **location**: `LOC`
* **channel_code** or **channel**: `CHA`
