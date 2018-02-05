Introduction
============

.. image:: images/logo/lasif_logo.*
    :width: 80%
    :align: center

**DISCLAIMER:** This documentation is still being updated. It is based on the
documentation of the previous version of LASIF and some parts might still
only apply to the previous version right now. We are working on updating
the documentation and once we feel confident that it is ready we will
remove this disclaimer.

LASIF 2.0 (**L**\ arg-scale **S**\ eismic **I**\ nversion **F**\ ramework) is a
**data-driven workflow tool** to perform full waveform inversions.  It is
opinionated and strict, meaning that it enforces a certain data and directory
structure. The advantage is that it only requires a very minimal amount of
configuration and maintenance. It attempts to gather all necessary information
from the data itself so there is no need to keep index or content files.

All parts of LASIF 2.0 also work completely on their own; see the class and
function documentation at the end of this document. Furthermore LASIF 2.0 offers a
project based inversion workflow management system which is introduced in the
following tutorial.

LASIF 2.0 works with the notion of so called inversion projects. A project is
defined as a series of iterations working on the same physical domain. Where
possible and useful, LASIF 2.0 will use hdf5 files and SQL databases to store
information. The reasoning behind this is twofold. The hdf5 files are of the
ASDF (Adaptable Seismic Data Format) which make it possible to reduce the amount
of files generally needed to store information and now all the information for
each event and each iteration can be stored in one file which helps with file
management. The SQL databases work very fast and are optimal for querying certain
parameters.

LASIF 2.0 is data-driven, meaning that it attempts to gather all necessary
information from the available data. The big advantage of this approach is that
the users can use any tool they want to access and work with the data as long
as they adhere to the directory structure imposed by LASIF. At the start of
every LASIF operation, the tool checks what data is available and uses it. To
achieve reasonable performance it employs a transparent caching scheme able to
quickly register any changes the user makes to the data. Also important to keep
in mind is that **LASIF 2.0 will never delete any data**.

The aim of the **LASIF 2.0** project is to facilitate the execution of mid-to
large-scale full seismic waveform inversion using adjoint techniques. A
simplified representation of the general workflow is presented here.
Right now these images are mostly related to LASIF 1.0 and don't necessarily
apply to LASIF 2.0. They will be updated soon.

.. image:: images/simplified_adjoint_workflow.*
    :width: 80%
    :align: center


One of the biggest problems is the meaningful organization of the different
types of data which are mostly in non-trivial relations to each other.


.. image:: images/LASIF_data_zoo.*
    :width: 80%
    :align: center


**LASIF 2.0** attempts to tackle these issues by employing a number of modules
tied together by a common project.

.. image:: images/LASIF_Overview.*
    :width: 80%
    :align: center


Supported Data Formats
----------------------

This is a short list of supported data formats and other software.


* **Waveform Data:** All file formats supported by ObsPy.
* **Synthetics:** All file formats supported by ObsPy and the output files of
  SALVUS
* **Event Metadata:** QuakeML 1.2
* **Station Metadata:** dataless SEED, RESP and FDSN StationXML. We strongly
  recommend to use StationXML!
* All these dataforms are put together and structured using pyasdf
* **Mesh Files:** Exodus mesh files. Recommended is the pymesher that works
    with SALVUS.
* **Waveform Solvers:** SALVUS


Further Notes
-------------

QuakeML files
^^^^^^^^^^^^^
LASIF 2.0 is designed to work with valid QuakeML 1.2 event files. Please assure
that the files you use actually are just that. If possible try to only use
QuakeML files with one origin and one focal mechanism, otherwise LASIF will
choose the preferred origin and/or focal mechanism (or the first of each, if no
preferred one is specified). **The origin time specified in the QuakeML file
will be the reference time for each event!** Times specified in SAC files will
be ignored.

This also means that the raw data files have to have the correct time
information.

Right now LASIF 2.0 works best if it is used to collect the events and the
relevant data. A tutorial on how to use your own data to produce a dataset
that LASIF 2.0 can work with will be on the website later.
