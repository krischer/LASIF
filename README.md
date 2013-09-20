![Logo](/doc/images/logo/lasif_logo.png)
---
[![Build Status](https://travis-ci.org/krischer/LASIF.png?branch=master)](https://travis-ci.org/krischer/LASIF)


> Work In Progress - Use With Caution


### Documentation and Tutorial

[Documentation and Tutorial](http://krischer.github.io/LASIF)

The documentation is always kept up-to-date with the master branch. It contains
a large tutorial intended to give an exhaustive tour of LASIF to aspiring new users.


### Installation

LASIF depends on a number of third-party libraries. These need to be installed
in order to enjoy all features of LASIF:

* [ObsPy](http://www.obspy.org) (in a very recent repository version - will be fixed once the next ObsPy version is released)
* The [matplotlib basemap toolkit](http://matplotlib.org/basemap/)
* [wfs_input_generator](http://github.com/krischer/wfs_input_generator)
* requests
* progressbar
* geographiclib
* colorama

The later four can simply be installed via pip:

```bash
$ pip install requests
$ pip install progressbar
$ pip install geographiclib
$ pip install colorama
```

ObsPy and basemap are slightly more complicated to install. Please refer to the
projects websites for more detailed instructions.

The wfs_input_generator can be installed with the following commands

```bash
$ git clone https://github.com/krischer/wfs_input_generator.git
$ cd wfs_input_generator
$ pip install -v -e .
$ cd ...
```

The actual LASIF module can then be installed with

```bash
$ git clone https://github.com/krischer/LASIF.git
$ cd LASIF
$ pip install -v -e .
$ cd ...
```

For running the tests, some additional modules are part of the requirements.
See the 'Testing' section below for more details.

Both, the `wfs_input_generator` and `LASIF` are in active development and thus
should be installed as a develop installation so they can be easily upgraded
via git.


### Testing

#### Requirements and Rational

LASIF evolved into a fairly complex piece of code and thus extensive testing is
required to assure that it works as expected.

To run the tests, you need to install four additional modules:

```bash
$ pip install pytest
$ pip install mock
$ pip install nose
$ pip install flake8
```

The [pytest](http://pytest.org) module is an alternative testing framework for
Python offering powerful test discovery features and a no-boilerplate approach
to syntax. It furthermore provides a nice functional approach to writing tests
for complex environments facilitating proper tests for LASIF.

Many operations in LASIF are computationally expensive or have side effects
like requiring online access making them not particularly well suited to being
testing. In order to verify some of the complex interactions within LASIF dummy
objects are used. These are provided by the
[mock](http://www.voidspace.org.uk/python/mock/) package.

LASIF contains some graphical functionality outputting maps and other plots.
matplotlib's testing facilities are reused in LASIF which in turn require the
[nose](http://nose.readthedocs.org/en/latest/) testing framework to be
installed.

The [flake8](http://flake8.readthedocs.org/en/2.0/) package is used to make
sure that LASIF's code base adhere to the
[PEP8](http://www.python.org/dev/peps/pep-0008/) standard.

#### Running the tests

To run the tests, cd somewhere into the LASIF project and type

```bash
$ py.test
```

This will recursively find and execute all tests below the current working
directory.

The py.test command accepts a large number of additional parameters, e.g.

```bash
# Execute only tests within test_project.py.
$ py.test test_project.py

# Print stdout and stderr and do not capture it.
$ py.test -s

# Execute only tests whose name contains the string 'some_string'.
$ py.test -k some_string
```

For more information please read the [pytest
documentation](http://pytest.org/).
