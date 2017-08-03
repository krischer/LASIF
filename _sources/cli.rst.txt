Command Line Interface
======================

The recommended way to interact with **LASIF** projects is to use the
command line interface. It consists of a number of subcommands grouped below
the main :code:`lasif` command. The general usage is quickly explained in this
paragraph whilst the remainder of this sections explains all subcommands in
greater detail.


The general help can be accessed with the :code:`lasif help` command.


.. runblock:: console
    :max_lines: 10

    $ lasif help


To access the subcommand specific help append :code:`--help` to the command.


.. runblock:: console

    $ lasif init_project --help



Each command can have a number of positional arguments and some optional
arguments. Positional arguments have to be given in the specified order
while optional arguments can be passed if needed.


.. note::

    All **lasif** commands work and use the correct project as long as they are
    executed somewhere **inside a project's folder structure**. It will
    recursively search the parent directories until it finds a
    :code:`config.xml` file. This will then be assumed to be the root folder
    of the project.


.. warning::

    LASIF employs a number of caches to keep it working fast enough.
    Information about each waveform, event, and station file for example is
    stored in the caches. The caches are usually automatically kept
    up-to-date: As soon as data is modified, deleted, added, ... the caches
    will know about it.

    Most commands have a ``--read_only_caches`` flag. If that argument is
    given, the caches will not be rebuilt but whatever is in the caches will
    be assumed to actually exist. This is useful when executing multiple
    LASIF commands in parallel. Otherwise some processes might write in
    parallel to the cash databases which will crash LASIF. Use this flag if you
    need to but best run ``$ lasif build_all_caches`` beforehand and be
    aware of what it means.

MPI
^^^

Some commands can be executed with MPI to speed up their execution. Don't
use too many cores as the problem quickly becomes I/O bounds. For example to
run the preprocessing on 16 cores, do

.. code-block:: bash

    $ mpirun -n 16 lasif preprocess_data 1 GCMT_event_AZORES_ISLANDS


The following commands are MPI-enabled. Attempting to run any other command
with MPI will result in an error.:

.. include_lasif_mpi_cli_commands::


Command Documentation
^^^^^^^^^^^^^^^^^^^^^

In the following all subcommands are documented in detail. The commands
are grouped by functionality.

.. contents:: Available Commands
    :local:
    :depth: 2

.. include_lasif_cli_commands::
