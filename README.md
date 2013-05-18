![Logo](/doc/logo/lasif_logo.png)

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

The later three can simply be installed via pip:

```bash
$ pip install requests
$ pip install progressbar
$ pip install geographiclib
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
