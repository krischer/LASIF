Data Validation
---------------

You might have noticed that LASIF projects can potentially contain many millions
of files and it will thus be impossible to validate the data by hand. LASIF
therefore contains a number of functions attempting to check the data of a
project. All of these can be called with

.. code-block:: bash

    $ lasif validate_data

Please make sure no errors appear -- otherwise LASIF cannot guarantee to work
correctly.  With time, more checks will be added to this function as more
problems arise.

This command will output a possibly very large report. All problems
indicated in it should be solved to ensure LASIF can operate correctly. The
best strategy to it is to start at the very beginning of the file. Often,
some errors are due to others and the *validate_data* command tries to be smart
and checks for simple errors first. By default, the command will launch only a
simplified validation function as some tests are rather slow. To get a full
validation going use

.. code-block:: bash

    $ lasif validate_data --full

Be aware that this may take a while.
