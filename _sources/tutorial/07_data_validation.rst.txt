.. centered:: Last updated on *August 12th 2016*.

Data Validation
---------------

You might have noticed that **LASIF** projects can potentially contain many millions
of files and it will thus be impossible to validate the data by hand. **LASIF**
therefore contains a number of functions attempting to check the data of a
project. All of these can be called with

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
        Validating against QuakeML 1.2 schema .. [OK]
        Checking for duplicate public IDs .. [OK]
        Performing some basic sanity checks .. [OK]
        Checking for duplicates and events too close in time ..  [OK]
        Assure all events are in chosen domain ..  [OK]
    Confirming that station metainformation files exist for all waveforms .. [OK]
    Checking all waveform files .. [OK]
    Making sure raypaths are within boundaries .. [OK]

    ALL CHECKS PASSED
    The data seems to be valid. If we missed something please contact the developers.

Be aware that this may take a while.

If **LASIF** detects something is amiss it will complain and potentially create
a script to remove the offending files. Review the script and execute it if
appropriate.
