Installation, Testing & DevInfo
===============================

This page details installation instructions, how to run the tests, and some
additional information if you want to develop new features for **LASIF 2.0**. If
you encounter any issues, please don't hesitate to contact us or
`open an issue on GitHub <https://github.com/dirkphilip/LASIF_2.0/issues/new>`_.


Installation
------------

Requirements
^^^^^^^^^^^^

**LASIF 2.0** is a Python based program and has a number of dependencies which
are listed here. It might well work with other versions but only the versions
listed here have been tested and are officially supported. It has been
tested on Linux and Mac OS X but should also run just fine on Windows.

If you know what you are doing, just make sure these dependencies are
available, otherwise please do yourself a favor and download the `Anaconda
<https://www.continuum.io/downloads>`_ Python distribution, a free package
containing almost all dependencies. Download it, install it, and follow the
upcoming instructions. It will install **LASIF** into a separate ``conda``
environment. This is very useful to separate the installation from the rest of
your system. Additionally it does not require root privileges and thus can be
installed almost everywhere.

.. code-block:: bash

    # Sometimes you need the newest version of conda to install packages
    $ conda update conda
    # Create a new conda environment which will here be called "lasif".
    $ conda create -n lasif python=3.6
    # Activate the lasif environment. This will always be needed when LASIF is started.
    $ source activate lasif
    # Start installing dependencies
    $ conda config --add channels conda-forge
    $ conda install -c conda-forge obspy nomkl basemap progressbar2 colorama joblib pytest nose mock pyqt
    $ conda install -c conda-forge pyqtgraph pip sphinx sphinx_rtd_theme numexpr ipython dill prov seaborn
    # Install more packages via pip
    $ pip install pyqtgraph geographiclib flask-cache geojson flake8 toml==0.9.2
    # Pick a directory where you want to store pyexodus and move into it
    $ git clone https://github.com/SalvusHub/pyexodus.git
    $ cd pyexodus
    $ pip install .
    # Make sure you do not have an active installation of mpi4py from conda
    $ conda uninstall mpi4py
    # re-install it using pip
    $ pip install mpi4py
    # Install a parallel version of hdf5
    $ conda install -c spectraldns h5py-parallel
    # Install pyasdf
    $ pip install pyasdf

Make sure that pyasdf is working as it should by running the following command

.. code-block:: bash

    $ python -c "import pyasdf; pyasdf.print_sys_info()"

.. code-block:: bash

    pyasdf version 0.3.0
    ===============================================================================
    CPython 3.6.4, compiler: GCC 4.2.1 Compatible Apple LLVM 6.1.0 (clang-602.0.53)
    Darwin 16.7.0 64bit
    Machine: x86_64, Processor: i386 with 4 cores
    ===============================================================================
    HDF5 version 1.8.14, h5py version: 2.6.0
    MPI: MPICH, version: 3.2.0, mpi4py version: 3.0.0
    Parallel I/O support: True
    Problematic multiprocessing: None
    ===============================================================================
    Other_modules:
	    dill: 0.2.7.1
	    lxml: 4.1.1
	    numpy: 1.14.0
	    obspy: 1.1.0

which should print something like the following lines:

We now need a component of `Salvus <https://www.salvus.io>`_ salvus seismo.

.. code-block:: bash

    # Move to a directory where you want to store salvus seismo and then execute
    $ git clone https://gitlab.com/Salvus/salvus_seismo
    $ cd salvus_seismo/py
    $ pip install -v -e .


Installing LASIF
^^^^^^^^^^^^^^^^

The actual **LASIF** module can then be installed with

.. code-block:: bash

    $ git clone https://github.com/dirkphilip/LASIF_2.0.git
    $ cd LASIF_2.0
    $ pip install -v -e .

After the installation one should run the tests to ensure everything is
installed correctly and works as intended on your machine.

Updating LASIF
^^^^^^^^^^^^^^

To update **LASIF 2.0**, change into the **LASIF 2.0** directory and type

.. code-block:: bash

    $ git pull
    $ pip install -v -e .


Testing
-------

**LASIF** evolved into a fairly complex piece of code and extensive testing is
required to assure that it works as expected.

Running the Tests
^^^^^^^^^^^^^^^^^

To run the tests, cd into the toplevel ``LASIF_2.0`` directory and execute:


.. code-block:: bash

    $ py.test

This will recursively find and execute all tests below the current working
directory.

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



Developer Information
---------------------


The following rules should be followed when developing for **LASIF 2.0**:

* **LASIF 2.0** is written entirely in Python 3.6.
* `Document <http://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/>`_ the
  code.
* Adhere to `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_.
* All contributed code must be contributed under the GPLv3.
* Write tests where reasonable.

    * **LASIF 2.0** utilizes `Travis CI <https://travis-ci.org/krischer/LASIF>`_
      for continuous integration testing. This means that every commit will be
      automatically tested and the responsible developer will receive an email
      in case her/his commit breaks **LASIF 2.0**.
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
