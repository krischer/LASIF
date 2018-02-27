Command Line Interface
======================

The recommended way to interact with **LASIF 2.0** projects is to use the
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
    :code:`lasif_config.toml` file. This will then be assumed to be the root folder
    of the project.

MPI
^^^

Some commands can be executed with MPI to speed up their execution. Don't
use too many cores as the problem quickly becomes I/O bounds. For example to
run the preprocessing on 16 cores, do

.. code-block:: bash

    $ mpirun -n 16 lasif process_data 1 GCMT_event_AZORES_ISLANDS


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
