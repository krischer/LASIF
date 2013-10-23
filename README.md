![Logo](/doc/images/logo/lasif_logo.png)
---
[![Build Status](https://travis-ci.org/krischer/LASIF.png?branch=master)](https://travis-ci.org/krischer/LASIF)
[![GPLv3](http://www.gnu.org/graphics/gplv3-88x31.png)](https://github.com/krischer/LASIF/blob/master/LICENSE)
> Work In Progress - Use With Caution


**LASIF** (Large-scale Seismic Inversion Framework) is a data-driven end-to-end
workflow tool to perform adjoint full seismic waveform inversions.

Dealing with the large amounts of data present in modern full seismic waveform
inversions in an organized, reproducible and shareable way continues to be a major
difficulty potentially even hindering actual research. **LASIF** improves the speed,
reliability, and ease with which such inversion can be carried out.

Full seismic waveform inversion using adjoint methods evolved into a well established
tool in recent years that has seen many applications. While the procedures employed
are (to a certain extent) well understood, large scale applications to real-world
problems are often hindered by practical issues.

The inversions use an iterative approach and thus by their very nature encompass
many repetitive, arduous, and error-prone tasks. Amongst these are data acquisition
and management, quality checks, preprocessing, selecting time windows suitable for
misfit calculations, the derivation of adjoint sources, model updates, and interfacing
with numerical wave propagation codes.

The **LASIF** workflow framework is designed to tackle these problems. One major focus of
the package is to handle vast amount of data in an organized way while also efficiently
utilizing modern HPC systems.
The use of a unified framework enables reproducibility and an efficient collaboration on
and exchange of tomographic images.


### Contents
* [Documentation and Tutorial](#documentation-and-tutorial)
* [Installation and Dependencies](#installation-and-dependencies)
* [Testing](#testing)
  - [Requirements and Rational](#requirements-and-rational)
  - [Running the Tests](#running-the-tests)
* [Developing for LASIF](#developing-for-lasif)
  - [Building the Documentation](#building-the-documentation)


### Documentation and Tutorial

[Documentation and Tutorial](http://krischer.github.io/LASIF)

The documentation is always kept up-to-date with the master branch. It contains
a large tutorial intended to give an exhaustive tour of **LASIF** to aspiring new users.


### Installation and Dependencies

**LASIF** depends on a number of third-party libraries. These need to be installed
in order to enjoy all features of **LASIF**:

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

The actual **LASIF** module can then be installed with

```bash
$ git clone https://github.com/krischer/LASIF.git
$ cd LASIF
$ pip install -v -e .
$ cd ...
```

For running the tests, some additional modules are part of the requirements.
See the 'Testing' section below for more details.

Both, the `wfs_input_generator` and **LASIF** are in active development and thus
should be installed as a develop installation so they can be easily upgraded
via git.


### Testing

#### Requirements and Rational

**LASIF** evolved into a fairly complex piece of code and thus extensive testing is
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
for complex environments facilitating proper tests for **LASIF**.

Many operations in **LASIF** are computationally expensive or have side effects
like requiring online access making them not particularly well suited to being
testing. In order to verify some of the complex interactions within **LASIF** dummy
objects are used. These are provided by the
[mock](http://www.voidspace.org.uk/python/mock/) package.

**LASIF** contains some graphical functionality outputting maps and other plots.
matplotlib's testing facilities are reused in **LASIF** which in turn require the
[nose](http://nose.readthedocs.org/en/latest/) testing framework to be
installed.

The [flake8](http://flake8.readthedocs.org/en/2.0/) package is used to make
sure that **LASIF**'s code base adhere to the
[PEP8](http://www.python.org/dev/peps/pep-0008/) standard.

#### Running the tests

To run the tests, cd somewhere into the **LASIF** code base and type

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


### Developing for LASIF

The following rules should be followed when developing for **LASIF**:

* **LASIF** is written entirely in Python.
* C/Fortran code with proper bindings can be used to improve performance where necessary.
* [Document](http://lukeplant.me.uk/blog/posts/docs-or-it-doesnt-exist/) the code.
* Adhere to [PEP8](http://www.python.org/dev/peps/pep-0008/#comments).
* All contributed code must be under the GPLv3.
* Write tests where reasonable.
  * **LASIF** utilizes [Travis CI](https://travis-ci.org/krischer/LASIF) for continuous
    integration testing. This means that every commit will be automatically tested and
    the responsible developer will receive an email in case her/his commit breaks **LASIF**.
  * The tests also verify the PEP8 conformance of the entire code base.

#### Building the Documentation

`sphinx` is used to build the documentation so it needs to be installed.

```bash
$ pip install sphinx
```

To actually build the documentation and save it in a subfolder:

```bash
$ cd doc
$ make html
```

#### Terminology

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
