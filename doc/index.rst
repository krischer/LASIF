.. SES3DPy documentation master file, created by
   sphinx-quickstart on Fri Feb  1 15:47:43 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SES3DPy's documentation!
===================================

Contents:

.. toctree::
   :maxdepth: 2

    rotations
    windows

Tutorial
========
SES3DPy works with the notion of inversion projects. A project is defined as a
series of iterations working on the same physical domain. Where possible and
useful, SES3DPy will use XML files to store information. The reasoning behind
this is twofold. It is easily machine and human readable. It also serves as a
preparatory steps towards a fully database driven full waveform inversion as
all necessary information is already stored in an easily indexable data format.

Creating a new Project
----------------------
The necessary first step, whether for starting a new inversion or migrating an
already existing inversion to SES3DPy, is to create a new project.

ses3dpy init_project MyInversion


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

