Welcome to LASIF
================

.. |br| raw:: html

    <br />

.. image:: images/logo/lasif_logo.*
    :width: 80%
    :align: center


LASIF (**LA**\ rge-scale **S**\ eismic **I**\ nversion **F**\ ramework) is a
data-driven end-to-end workflow tool to perform adjoint full seismic waveform
inversions.

**Github:** The code is developed at and available from its
`Github respository <http://github.com/krischer/LASIF>`_.


.. admonition:: For more details please see our paper

    *Lion Krischer, Andreas Fichtner, Saule Zukauskaite, and Heiner Igel* (2015), |br|
    **Large‐Scale Seismic Inversion Framework**, |br|
    Seismological Research Letters, 86(4), 1198–1207. |br|
    `doi:10.1785/0220140248 <http://dx.doi.org/10.1785/0220140248>`_

---------


Dealing with the large amounts of data present in modern full seismic waveform
inversions in an organized, reproducible and shareable way continues to be a
major difficulty potentially even hindering actual research. LASIF improves the
speed, reliability, and ease with which such inversion can be carried out.

Full seismic waveform inversion using adjoint methods evolved into a well
established tool in recent years that has seen many applications. While the
procedures employed are (to a certain extent) well understood, large scale
applications to real-world problems are often hindered by practical issues.

The inversions use an iterative approach and thus by their very nature
encompass many repetitive, arduous, and error-prone tasks. Amongst these are
data acquisition and management, quality checks, preprocessing, selecting time
windows suitable for misfit calculations, the derivation of adjoint sources,
model updates, and interfacing with numerical wave propagation codes.

The LASIF workflow framework is designed to tackle these problems. One major
focus of the package is to handle vast amount of data in an organized way while
also efficiently utilizing modern HPC systems. The use of a unified framework
enables reproducibility and an efficient collaboration on and exchange of
tomographic images.


.. toctree::
    :hidden:
    :maxdepth: 2

    prerequisites

    introduction
    tutorial

    cli
    webinterface
    api_doc
    faq

