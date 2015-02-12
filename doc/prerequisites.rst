Installation, Testing & DevInfo
===============================

This page details installation instructions, how to run the tests, and some
additional information if you want to develop new features for **LASIF**. If
you encounter any issues, please don't hesitate to contact us or
`open an issue on GitHub <https://github.com/krischer/LASIF/issues/new>`_.


Installation
------------

Requirements
^^^^^^^^^^^^

**LASIF** is a Python based program and has a number of dependencies which
are listed here. It might well work with other versions but only the versions
listed here have been tested and are officially supported. It has been
tested on Linux and Mac OS X but should also run just fine on Windows.

* ``obspy`` (`www.obspy.org <http://www.obspy.org/>`_ - in a recent
  respository version)
* ``numpy >= 1.8``
* ``matplotlib >= 1.3``
* ``basemap >= 1.0.7``
* ``wfs_input_generator`` (`website <http://github.com/krischer/wfs_input_generator>`_)
* ``geographiclib``
* ``progressbar`` and ``colorama``
* ``mpi4py`` and ``joblib``
* *For the webinterface, you additionally need:* ``flask``, ``flask-cache``,
  *and* ``geojson``.
* *The misfit GUI requires* ``pyqt`` *and* ``pyqtgraph``.
* *And for running the tests you need* ``pytest``, ``mock``, ``nose``, *and*
  ``flake8``.


If you know what you are doing, just make sure these dependencies are
available, otherwise please do yourself a favor and download the
`Anaconda <https://store.continuum.io/cshop/anaconda/>`_ Python
distribution, a free package containing almost all dependencies. Make sure
to choose the Python 2.7 version. Install ObsPy according to
`these instructions <https://github.com/obspy/obspy/wiki/Installation-via-Anaconda>`_
but do not execute ``pip install obspy``. Instead install via git:

.. code-block:: bash

    $ git clone https://github.com/obspy/obspy.git
    $ cd obspy
    $ pip install -v -e .

This will install the latest repository version of ObsPy. At some point in
the future the latest stable ObsPy version will also be sufficient but right
now **LASIF** depends on features not yet in the latest stable ObsPy version.
Install the remaining dependencies with

.. code-block:: bash

    $ conda install numpy matplotlib basemap mpi4py flask pyqt pytest mock nose flake8
    $ pip install geographiclib progressbar colorama joblib flask-cache geojson pyqtgraph

Finally install the *wfs_input_generator* module again via git:

.. code-block:: bash

    $ git clone https://github.com/krischer/wfs_input_generator.git
    $ cd wfs_input_generator
    $ pip install -v -e .


Installing LASIF
^^^^^^^^^^^^^^^^

The actual **LASIF** module can then be installed with

.. code-block:: bash

    $ git clone https://github.com/krischer/LASIF.git
    $ cd LASIF
    $ pip install -v -e .

After the installation one should run the tests to ensure everything is
installed correctly and works as intended on your machine.


Testing
-------

**LASIF** evolved into a fairly complex piece of code and extensive testing is
required to assure that it works as expected.

Running the Tests
^^^^^^^^^^^^^^^^^

To run the tests, cd into the toplevel ``LASIF`` directory and execute:


.. code-block:: bash

    $ py.test

This will recursively find and execute all tests below the current working
directory.


If your machine has multiple cores, the processing can also be sped up
quite a bit by using ``pytest-xdist`` which can be installed via pip.

.. code-block:: bash

    $ pip install pytest-xdist

It enables to distribute the tests across cores. To run on, for example, eight
cores, use

.. code-block:: bash

    $ py.test -n 8


Building the Documentation
--------------------------

``sphinx`` is used to build the documentation so it needs to be installed. The
theme is the standalone **readthedocs** theme. We will use the most up-to-date
repository version here.

.. code-block:: bash

    $ pip install sphinx
    $ pip install https://github.com/snide/sphinx_rtd_theme/archive/master.zip

To actually build the documentation (in this case in the HTML format), run

.. code-block:: bash

    $ cd doc
    $ make html

This might take a while if run for the first time. Subsequent runs are faster.


Developer Information
---------------------


The following rules should be followed when developing for **LASIF**:

* **LASIF** is written entirely in Python 2.7. Adding support for 3.x would
  not be a big issue if necessary.
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
