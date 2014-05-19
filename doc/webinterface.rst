Web Interface
=============

The web interface is way to interactively explore a LASIF project and the
data within. To start it, simply execute the following inside the folder
structure of a LASIF project:

.. code-block:: bash

    $ lasif serve


This will start a local web server and open the website in your browser. It
heavily relies on Javascript so make sure it is not disabled. Please also use
a modern and up-to-date browser. I tested it with Firefox, Chrome, and Safari;
Firefox being the slowest of the bunch.

The webinterface is designed to display the state of a project at the time
the ``serve`` command is executed. If you modify the state of a project
while the web server is running, the result is undefined. Please restart the
server in that case.


.. image:: images/webinterface/1.png
    :width: 100%
    :align: center

.. image:: images/webinterface/2.png
    :width: 100%
    :align: center

.. image:: images/webinterface/3.png
    :width: 100%
    :align: center

.. image:: images/webinterface/4.png
    :width: 100%
    :align: center

.. image:: images/webinterface/5.png
    :width: 100%
    :align: center
