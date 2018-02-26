.. centered:: Last updated on *February 22nd 2018*.

Station Data
------------

**LASIF** needs to know the coordinates and instrument response of each channel.
The best way to achieve this is to have a StationXML file related to each
station and each event. These StationXML files then have to be read into the
ASDF file containing the Earthquake information. We will make a tutorial on
how to organize your files into an ASDF format. Hopefully we will have time
to do that soon. But for now we recommend using the built in downloading
schemes which will be explained soon.

Waveform Data
^^^^^^^^^^^^^

Every inversion needs real data to be able to quantify misfits. In **LASIF**
this data needs to be stored, as we have mentioned several times before, in
the ASDF file containing Earthquake information. These files are positioned
under the name: *{project_root}/DATA/EARTHQUAKES/{event_name}.h5*. Only one
file is used to contain all the raw information for each event. Including
earthquake information (QuakeML), Station information (StationXML) and raw
waveforms for each station.