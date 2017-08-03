.. centered:: Last updated on *August 12th 2016*.

Data Inspection
---------------

Once waveform and station metadata has been downloaded (either with the
built-in helpers or manually) and placed in the correct folders, **LASIF** can
start to work with it.

.. note::

    **LASIF** essentially needs three ingredients to be able to interpret waveform
    data:

    * The actual waveforms
    * The location of the recording seismometer
    * The instrument response for each channel at the time of data recording

    Some possibilities exist to specify these:

    * MiniSEED and StationXML (strongloy preferred)
    * SAC data and RESP files (needed for legacy reasons)
    * MiniSEED and RESP files (this combination does not actually contain
      location information but **LASIF** launches some web requests to get just the
      locations and stores them in a cache database)
    * Most other combinations should also work but have not been tested.


At this point, **LASIF** is able to match available station and waveform
information. Only stations where the three aforementioned ingredients are
available will be considered to be stations that are good to be worked with by
**LASIF**. Others will be ignored.

To get an overview, of what data is actually available for any given event,
just execute:

.. code-block:: bash

    $ lasif event_info -v GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17

    Earthquake with 4.9 Mwc at NORTHERN ITALY
	Latitude: 44.870, Longitude: 8.480, Depth: 15.0 km
	2000-08-21T17:14:31.100000Z UTC

    Station and waveform information available at 51 stations:

    ===========================================================================
             id       latitude      longitude elevation_in_m    local depth
    ===========================================================================
        LA.AA00  44.4567283802  5.62467939102            0.0            0.0
        LA.AA01   42.192951649  23.3492243512            0.0            0.0
        LA.AA02  53.2426492521  16.0639363825            0.0            0.0
        LA.AA03  34.5585730668   9.5857134452            0.0            0.0
        LA.AA04  45.6821185764  15.1965770125            0.0            0.0
        LA.AA05  50.2345625685  9.03513474669            0.0            0.0
        ...

.. note::

    The local depth is allowed to not be set. In that case it will be
    assumed to be zero. For all practical purposes the local depth does not
    matter for continental scale inversions.


It is furthermore possible to plot the availability information for one event
including a very simple ray coverage plot with:

.. code-block:: bash

    $ lasif plot_event GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17


.. plot::

    import matplotlib.pylab as plt
    from obspy import UTCDateTime
    from lasif import domain
    import lasif.visualization

    bmap = domain.RectangularSphericalSection(
        min_latitude=-10,
        max_latitude=10,
        min_longitude=-10,
        max_longitude=10,
        min_depth_in_km=0,
        max_depth_in_km=1440,
        boundary_width_in_degree=2.5,
        rotation_axis=[1.0, 1.0, 0.2],
        rotation_angle_in_degree=-65.0).plot(plot_simulation_domain=False)
    event_info = {'depth_in_km': 10.0,
        'event_name': 'GCMT_EVENT_NORTHERN_ITALY_Mag_4.9_2000-8-21-17',
        'filename': 'GCMT_EVENT_NORTHERN_ITALY_Mag_4.9_2000-8-21-17.xml',
        'latitude': 44.87, 'longitude': 8.48, 'm_pp': 1.189e+16,
        'm_rp': -1600000000000000.0, 'm_rr': -2.271e+16,
        'm_rt': -100000000000000.0, 'm_tp': -2.075e+16, 'm_tt': 1.082e+16,
        'magnitude': 4.9, 'magnitude_type': 'Mwc',
        'origin_time': UTCDateTime(2000, 8, 21, 17, 14, 27),
        'region': u'NORTHERN ITALY'}

    stations = {u'LA.AA00': {'elevation_in_m': 0.0,
              'latitude': 44.4567283802,
              'local_depth_in_m': 0.0,
              'longitude': 5.62467939102},
         u'LA.AA01': {'elevation_in_m': 0.0, 'latitude': 42.192951649,
                      'local_depth_in_m': 0.0,
                      'longitude': 23.3492243512},
         u'LA.AA02': {'elevation_in_m': 0.0,
                      'latitude': 53.2426492521,
                      'local_depth_in_m': 0.0,
                      'longitude': 16.0639363825},
         u'LA.AA03': {'elevation_in_m': 0.0,
                      'latitude': 34.5585730668,
                      'local_depth_in_m': 0.0,
                      'longitude': 9.5857134452},
         u'LA.AA04': {'elevation_in_m': 0.0,
                      'latitude': 45.6821185764,
                      'local_depth_in_m': 0.0,
                      'longitude': 15.1965770125},
         u'LA.AA05': {'elevation_in_m': 0.0,
                      'latitude': 50.2345625685,
                      'local_depth_in_m': 0.0,
                      'longitude': 9.03513474669},
         u'LA.AA06': {'elevation_in_m': 0.0,
                      'latitude': 39.0566403496,
                      'local_depth_in_m': 0.0,
                      'longitude': 16.2628129402},
         u'LA.AA07': {'elevation_in_m': 0.0,
                      'latitude': 40.3377603385,
                      'local_depth_in_m': 0.0,
                      'longitude': 9.24702378562},
         u'LA.AA08': {'elevation_in_m': 0.0,
                      'latitude': 45.3001671698,
                      'local_depth_in_m': 0.0,
                      'longitude': -0.357405368137},
         u'LA.AA09': {'elevation_in_m': 0.0,
                      'latitude': 46.803809547,
                      'local_depth_in_m': 0.0,
                      'longitude': 22.2985397715},
         u'LA.AA10': {'elevation_in_m': 0.0,
                      'latitude': 41.3317000452,
                      'local_depth_in_m': 0.0,
                      'longitude': 2.00073761549},
         u'LA.AA11': {'elevation_in_m': 0.0,
                      'latitude': 49.2089062992,
                      'local_depth_in_m': 0.0,
                      'longitude': 14.4358999924},
         u'LA.AA12': {'elevation_in_m': 0.0,
                      'latitude': 42.2427301565,
                      'local_depth_in_m': 0.0,
                      'longitude': 13.7642758663},
         u'LA.AA13': {'elevation_in_m': 0.0,
                      'latitude': 48.5108717569,
                      'local_depth_in_m': 0.0,
                      'longitude': 4.02709492648},
         u'LA.AA14': {'elevation_in_m': 0.0,
                      'latitude': 39.0615631384,
                      'local_depth_in_m': 0.0,
                      'longitude': 20.8596848758},
         u'LA.AA15': {'elevation_in_m': 0.0,
                      'latitude': 46.8385400359,
                      'local_depth_in_m': 0.0,
                      'longitude': 10.1338319588},
         u'LA.AA16': {'elevation_in_m': 0.0,
                      'latitude': 43.4761324632,
                      'local_depth_in_m': 0.0,
                      'longitude': 19.5514592756},
         u'LA.AA17': {'elevation_in_m': 0.0,
                      'latitude': 37.1214874112,
                      'local_depth_in_m': 0.0,
                      'longitude': 9.27888297788},
         u'LA.AA18': {'elevation_in_m': 0.0,
                      'latitude': 37.9552427568,
                      'local_depth_in_m': 0.0,
                      'longitude': 13.0850518087},
         u'LA.AA19': {'elevation_in_m': 0.0,
                      'latitude': 49.9317849832,
                      'local_depth_in_m': 0.0,
                      'longitude': 19.5553707429},
         u'LA.AA20': {'elevation_in_m': 0.0,
                      'latitude': 43.113385089,
                      'local_depth_in_m': 0.0,
                      'longitude': 10.5025122695},
         u'LA.AA21': {'elevation_in_m': 0.0,
                      'latitude': 52.1619375637,
                      'local_depth_in_m': 0.0,
                      'longitude': 12.5756968471},
         u'LA.AA22': {'elevation_in_m': 0.0,
                      'latitude': 47.4445709809,
                      'local_depth_in_m': 0.0,
                      'longitude': 18.0006996813},
         u'LA.AA23': {'elevation_in_m': 0.0,
                      'latitude': 48.2242466203,
                      'local_depth_in_m': 0.0,
                      'longitude': 7.52362903015},
         u'LA.AA24': {'elevation_in_m': 0.0,
                      'latitude': 43.7482710675,
                      'local_depth_in_m': 0.0,
                      'longitude': 16.6768049734},
         u'LA.AA25': {'elevation_in_m': 0.0,
                      'latitude': 44.496175607,
                      'local_depth_in_m': 0.0,
                      'longitude': 24.1364065368},
         u'LA.AA26': {'elevation_in_m': 0.0,
                      'latitude': 45.0711271767,
                      'local_depth_in_m': 0.0,
                      'longitude': 12.0850673762},
         u'LA.AA27': {'elevation_in_m': 0.0,
                      'latitude': 40.7581463635,
                      'local_depth_in_m': 0.0,
                      'longitude': 19.3779870971},
         u'LA.AA28': {'elevation_in_m': 0.0,
                      'latitude': 41.404992886,
                      'local_depth_in_m': 0.0,
                      'longitude': 26.1572187768},
         u'LA.AA29': {'elevation_in_m': 0.0,
                      'latitude': 43.5655196629,
                      'local_depth_in_m': 0.0,
                      'longitude': 1.78107240944},
         u'LA.AA30': {'elevation_in_m': 0.0,
                      'latitude': 46.5848784846,
                      'local_depth_in_m': 0.0,
                      'longitude': 5.3054128683},
         u'LA.AA31': {'elevation_in_m': 0.0,
                      'latitude': 45.443324266,
                      'local_depth_in_m': 0.0,
                      'longitude': 8.09910575421},
         u'LA.AA32': {'elevation_in_m': 0.0,
                      'latitude': 45.6079198439,
                      'local_depth_in_m': 0.0,
                      'longitude': 19.0895934123},
         u'LA.AA33': {'elevation_in_m': 0.0,
                      'latitude': 45.8679517353,
                      'local_depth_in_m': 0.0,
                      'longitude': 2.73863085813},
         u'LA.AA34': {'elevation_in_m': 0.0,
                      'latitude': 51.1366867607,
                      'local_depth_in_m': 0.0,
                      'longitude': 16.4095614983},
         u'LA.AA35': {'elevation_in_m': 0.0,
                      'latitude': 41.8925113504,
                      'local_depth_in_m': 0.0,
                      'longitude': 16.0357355172},
         u'LA.AA36': {'elevation_in_m': 0.0,
                      'latitude': 47.3283608837,
                      'local_depth_in_m': 0.0,
                      'longitude': 13.2299336001},
         u'LA.AA37': {'elevation_in_m': 0.0,
                      'latitude': 39.9418567957,
                      'local_depth_in_m': 0.0,
                      'longitude': 22.6153264114},
         u'LA.AA38': {'elevation_in_m': 0.0,
                      'latitude': 51.8433876344,
                      'local_depth_in_m': 0.0,
                      'longitude': 9.98397600097},
         u'LA.AA39': {'elevation_in_m': 0.0,
                      'latitude': 36.7965784424,
                      'local_depth_in_m': 0.0,
                      'longitude': 14.8788013231},
         u'LA.AA40': {'elevation_in_m': 0.0,
                      'latitude': 43.0719575651,
                      'local_depth_in_m': 0.0,
                      'longitude': -0.364658357725},
         u'LA.AA41': {'elevation_in_m': 0.0,
                      'latitude': 44.7131302431,
                      'local_depth_in_m': 0.0,
                      'longitude': 21.4859649817},
         u'LA.AA42': {'elevation_in_m': 0.0,
                      'latitude': 49.8025093099,
                      'local_depth_in_m': 0.0,
                      'longitude': 6.57805436818},
         u'LA.AA43': {'elevation_in_m': 0.0,
                      'latitude': 49.0039392934,
                      'local_depth_in_m': 0.0,
                      'longitude': 11.0395191447},
         u'LA.AA44': {'elevation_in_m': 0.0,
                      'latitude': 48.323662543,
                      'local_depth_in_m': 0.0,
                      'longitude': 20.5836296776},
         u'LA.AA45': {'elevation_in_m': 0.0,
                      'latitude': 49.2292369799,
                      'local_depth_in_m': 0.0,
                      'longitude': 16.9815826589},
         u'LA.AA46': {'elevation_in_m': 0.0,
                      'latitude': 42.7884054523,
                      'local_depth_in_m': 0.0,
                      'longitude': 25.1671809184},
         u'LA.AA47': {'elevation_in_m': 0.0,
                      'latitude': 41.6345417449,
                      'local_depth_in_m': 0.0,
                      'longitude': 21.3579940528},
         u'LA.AA48': {'elevation_in_m': 0.0,
                      'latitude': 36.8710108404,
                      'local_depth_in_m': 0.0,
                      'longitude': 7.16817986685},
         u'LA.AA49': {'elevation_in_m': 0.0,
                      'latitude': 35.5740640944,
                      'local_depth_in_m': 0.0,
                      'longitude': 11.0359996403},
         u'LA.AA50': {'elevation_in_m': 0.0,
                      'latitude': 50.6507532779,
                      'local_depth_in_m': 0.0,
                      'longitude': 11.8025058982}}
    lasif.visualization.plot_stations_for_event(map_object=bmap,
        station_dict=stations, event_info=event_info)
    lasif.visualization.plot_events([event_info], bmap)
    plt.show()


If you are interested in getting a coverage plot of all events and data
available for the current project, please execute the ``plot_raydensity``
command:

.. code-block:: bash

    $ lasif plot_raydensity

Actually plotting this may take a fair while, depending on the amount of data
you have.
Keep in mind that this only results in a reasonable plot for large amounts of
data; for the toy example used in the tutorial it will not work. It is not a
physically accurate plot but helps in judging data coverage and directionality
effects. An example from a larger **LASIF** project illustrates this:


.. image:: ../images/raydensity.jpg
    :width: 70%
    :align: center



