Introduction
============

.. image:: images/logo/lasif_logo.*
    :width: 80%
    :align: center


LASIF (**L**\ arg-scale **S**\ eismic **I**\ nversion **F**\ ramework) is a
**data-driven workflow tool** to perform full waveform inversions.  It is
opinionated and strict meaning that it enforces a certain data and directory
structure. The upside is that it only requires a very minimal amount of
configuration and maintenance. It attempts to gather all necessary information
from the data itself so there is no need to keep index or content files.

All parts of LASIF also work completely on their own; see the class and
function documentation at the end of this document. Furthermore LASIF offers a
project based inversion workflow management system which is introduced in the
following tutorial.

LASIF works with the notion of so called inversion projects. A project is
defined as a series of iterations working on the same physical domain. Where
possible and useful, LASIF will use XML files to store information. The
reasoning behind this is twofold. It is easily machine and human readable. It
furthermore also serves as a preparatory step towards a fully database driven
full waveform inversion workflow as all necessary information is already stored
in an easily indexable data format.

LASIF is data-driven meaning that it attempts to gather all necessary
information from the available data. The big advantage of this approach is that
the users can use any tool they want to access and work with the data as long
as they adhere to the directory structure imposed by LASIF. At the start of
every LASIF operation, the tool checks what data is available and uses it. To
achieve reasonable performance it employs a transparent caching scheme able to
quickly any changes the user makes to the data. Also important to keep in mind
is that **LASIF will never delete any data**.

The aim of the **LASIF** project is to facilitate the execution of mid-to
large-scale full seismic waveform inversion using adjoint techniques. A
simplified representation of the general workflow is presented here.

.. image:: images/simplified_adjoint_workflow.*
    :width: 80%
    :align: center


One of the biggest problems is the meaningful organization of the different
types of data which are mostly in non-trivial relations to each other.


.. image:: images/LASIF_data_zoo.*
    :width: 80%
    :align: center


**LASIF** attempts to tackle these issues by employing a number of modules
tied together by a common project.

.. image:: images/LASIF_Overview.*
    :width: 80%
    :align: center

Further Information
-------------------

The documentation is currently being restructures. For now additional
information can be found here:

* :doc:`how_lasif_finds_coordinates`


TO DO: Things that are still missing
------------------------------------

This is mainly useful for the developers.

* Applying the time corrections for events and stations
* Settings for the time frequency misfit
* A clean way to integrate other misfits
* Data rejection criteria implementation
* Log more things for better provenance


Supported Data Formats
----------------------

This is a short list of supported data formats and other software.


* **Waveform Data:** All file formats supported by ObsPy.
* **Synthetics:** All file formats supported by ObsPy and the output files of
  SES3D 4.1.
* **Event Metadata:** QuakeML 1.2
* **Station Metadata:** dataless SEED, RESP and (hopefully) soon FDSN
  StationXML.  Once implemented, StationXML will be the recommended and most
  future proof format.
* **Earth Models:** Currently the raw SES3D model format is supported.
* **Waveform Solvers:** SES3D 4.1, support for SpecFEM Cartesian and/or Globe
  will be added soon.


Further Notes
-------------

QuakeML files
^^^^^^^^^^^^^
LASIF is designed to work with valid QuakeML 1.2 event files. Please assure
that the files you use actually are just that. If possible try to only use
QuakeML files with one origin and one focal mechanism, otherwise LASIF will
choose the preferred origin and/or focal mechanism (or the first of each, if no
preferred one is specified). **The origin time specified in the QuakeML file
will be the reference time for each event!** Times specified in SAC files will
be ignored.

This also means that the raw data files have to have the correct time
information.

Synthetic Data Files
^^^^^^^^^^^^^^^^^^^^
The very first sample of each synthetic waveform will be assumed to coincide
with the event time. If this is not a reasonable assumption, please contact the
LASIF developers.


