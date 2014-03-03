CLI Interface
=============

LASIF ships with a command line interface, consisting of a single command:
**lasif**.

Assuming the installation was successful, the following command will print a
short overview of all commands available within LASIF:

.. code-block:: bash

    $ lasif --help

    Usage: lasif FUNCTION PARAMETERS

    Available functions:
        add_spud_event
        create_new_iteration
        ...

To learn more about a specific command, append *--help* to it:

.. code-block:: bash

    $ lasif init_project --help

    Usage: lasif init_project FOLDER_PATH

        Creates a new LASIF project at FOLDER_PATH. FOLDER_PATH must not exist
        yet and will be created.


.. note::

    All **lasif** commands work and use the correct project as long as they are
    executed somewhere inside a project's folder structure. It will recursively
    search the parent directories until it finds a *config.xml* file. This will
    then be assumed to be the root folder of the project.


.. include_lasif_cli_commands::
