![Logo](/doc/logo/lasif_logo.png)

---
---

> Work In Progress - Use With Caution

---
---

[Preliminary Documentation](http://krischer.github.io/LASIF)


### Installation

LASIF depends on [ObsPy](http://www.obspy.org) in a recent version.
Furthermore it requires a waveform solver input file generator. This
can be installed with

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
