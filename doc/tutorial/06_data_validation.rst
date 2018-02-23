.. centered:: Last updated on *February 23rd 2018*.

Data Validation
---------------

You might have noticed that **LASIF** projects can potentially contain files
which are very large and it will thus be impossible to validate the data by
hand. **LASIF** therefore contains a number of functions attempting to
check the data of a project. All of these can be called with

.. code-block:: bash

    $ lasif validate_data

Please make sure no errors appear -- otherwise **LASIF** cannot guarantee to work
correctly.  With time, more checks will be added to this function as more
problems arise.

This command will output a possibly very large report. All problems
indicated in it should be solved to ensure **LASIF** can operate correctly. The
best strategy to it is to start at the very beginning of the file. Often,
some errors are due to others and the ``validate_data`` command tries to be smart
and checks for simple errors first. By default, the command will launch only a
simplified validation function as some tests are rather slow. To get a full
validation going use

.. code-block:: bash

    $ lasif validate_data --full

        Validating 2 event files ...
        Performing some basic sanity checks .. [OK]
        Checking for duplicates and events too close in time .. [OK]
        Assure all events are in chosen domain .. [OK]
    Confirming that station metainformation files exist for all waveforms .. [OK]
    Making sure raypaths are within boundaries .. [OK]

    ALL CHECKS PASSED
    The data seems to be valid. If we missed something please contact the developers.
    Be aware that this may take a while.

If **LASIF** detects something is amiss it will complain and potentially create
a script to remove the offending files. Review the script and execute it if
appropriate. *This is not working right now but will maybe be added later*.

It is possible to delete stations manually using pyasdf commands or by using
the misfit gui which will be explained later.
