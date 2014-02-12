Developer Information
=====================


The following rules should be followed when developing for **LASIF**:

* **LASIF** is written entirely in Python.
* C/Fortran code with proper bindings can be used to improve performance where
  necessary. Cython is also an accepted alternative.
* `Document <http://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/>`_ the
  code.
* Adhere to `PEP8 <http://www.python.org/dev/peps/pep-0008/#comments>`_.
* All contributed code must be contributed under the GPLv3.
* Write tests where reasonable.

    * **LASIF** utilizes `Travis CI <https://travis-ci.org/krischer/LASIF>`_
      for continuous integration testing. This means that every commit will be
      automatically tested and the responsible developer will receive an email
      in case her/his commit breaks **LASIF**.
    * The tests also verify the PEP8 conformance of the entire code base.

Building the Documentation
--------------------------

`sphinx` is used to build the documentation so it needs to be installed.

.. code-block:: bash

    $ pip install sphinx

To actually build the documentation (in this case in the HTML format), run

.. code-block:: bash

    $ cd doc
    $ make html

Terminology
-----------

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
