Testing
=======

Requirements and Rational
-------------------------

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
-----------------

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
