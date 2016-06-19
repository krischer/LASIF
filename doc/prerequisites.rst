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

* ``obspy >= 1.0.1`` (`www.obspy.org <http://www.obspy.org/>`_)
* ``numpy >= 1.8``
* ``numexpr`` (newer versions are also oftentimes faster)
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
available, otherwise please do yourself a favor and download the `Anaconda
<https://www.continuum.io/downloads>`_ Python distribution, a free package
containing almost all dependencies. Download it, install it, and follow the
upcoming instructions. It will install **LASIF** into a separate ``conda``
environment. This is very useful to separate the installation from the rest of
your system. Additionally it does not require root privileges and thus can be
installed almost everywhere.

.. code-block:: bash

    # Create a new conda environment named "lasif".
    $ conda create -n lasif python=2.7
    # Activate that environment. You will have to do this every time you start LASIF.
    # If you don't want to do that: put that line in your .bashrc/.bash_profile
    $ source activate lasif
    # Install most things via conda.
    $ conda install -c obspy obspy basemap progressbar colorama joblib flask pyqt pyqtgraph pytest nose mock flake8 pip numexpr
    # Install some missing things over pip.
    $ pip install geographiclib flask-cache geojson
    # Install the wfs_input_generator package.
    $ pip install https://github.com/krischer/wfs_input_generator/archive/master.zip

Installing LASIF
^^^^^^^^^^^^^^^^

The actual **LASIF** module can then be installed with

.. code-block:: bash

    $ git clone https://github.com/krischer/LASIF.git
    $ cd LASIF
    $ pip install -v -e .

After the installation one should run the tests to ensure everything is
installed correctly and works as intended on your machine.

Updating LASIF
^^^^^^^^^^^^^^

To update **LASIF**, change into the **LASIF** directory and type

.. code-block:: bash

    $ git pull
    $ pip install -v -e .

Additionally you might have to update the `wfs_input_generator`:

.. code-block:: bash

    $ pip install https://github.com/krischer/wfs_input_generator/archive/master.zi

Please note, that updating **LASIF** will not update your custom, user-defined
functions within your projects (see :doc:`tutorial/16_customizing_lasif`).  If
you want to update those as well: delete them and execute any **LASIF**
function - this will copy the latest versions of these files to your project
directory. **Make sure to save any changes you made to those functions!!**


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
directory. The output should look akin to the following:

.. code-block:: bash

    ===================================== test session starts =====================================
    platform darwin -- Python 2.7.11, pytest-2.9.1, py-1.4.31, pluggy-0.3.1
    rootdir: /Users/lion/workspace/code/LASIF, inifile: pytest.ini
    collected 195 items

    lasif/rotations.py .......
    lasif/utils.py ..
    lasif/window_selection.py .
    ...

    ===================== 189 passed, 4 skipped, 2 xfailed in 160.14 seconds ======================

No errors should occur. **If you see nothing - make sure your MPI installation
is correct (see above)**.

Assuming your machine has multiple cores, the test can also be sped up
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

    $ conda install sphinx sphinx_rtd_theme

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
