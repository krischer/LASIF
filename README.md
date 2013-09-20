![Logo](/doc/images/logo/lasif_logo.png)
---
[![Build Status](https://travis-ci.org/krischer/LASIF.png?branch=master)](https://travis-ci.org/krischer/LASIF)

---
---

> Work In Progress - Use With Caution

---
---

### Documentation

[Preliminary Documentation](http://krischer.github.io/LASIF)

This is not up-to-date and will be replaced with a new version soon. Hang tight!

To locally build the most recent version:

```bash
$ cd doc
$ make html
```


### Installation

Dependencies:

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


### Testing

The test are written with the pytest framework and require four additional modules:

```bash
$ pip install pytest
$ pip install mock
$ pip install nose
$ pip install flake8
```

The `mock` module is used for testing the command line interface.

The `nose` module is required for the image comparison tests which leverage
matplotlib's testing facilities which in turn require nose to run.

To run the test, cd to into the LASIF project and type

```bash
$ py.test
```

Many more options for testing are available. Please read the [pytest
documentation](http://pytest.org/) for more information.
