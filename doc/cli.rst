CLI Interface
=============

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


In the following all subcommands are documented in detail. The commands
are grouped by functionality.


.. include_lasif_cli_commands::
