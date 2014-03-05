import matplotlib.pylab as plt
from obspy import UTCDateTime
import lasif.visualization
map = lasif.visualization.plot_domain(34.1, 42.9, 23.1, 42.9, 1.46,
    rotation_axis=[0.0, 0.0, 1.0], rotation_angle_in_degree=0.0,
    zoom=True)
event_info = {'depth_in_km': 4.5, 'region': 'TURKEY', 'longitude': 40.14,
    'magnitude': 5.1, 'magnitude_type': 'Mwc', 'latitude': 38.82,
    'origin_time': UTCDateTime(2010, 3, 24, 14, 11, 31)}
stations = {u'GE.APE': {'latitude': 37.0689, 'local_depth': 0.0,
    'elevation': 620.0, 'longitude': 25.5306}, u'HL.ARG': {'latitude':
    36.216, 'local_depth': 0.0, 'elevation': 170.0, 'longitude': 28.126},
    u'IU.ANTO': {'latitude': 39.868, 'local_depth': None, 'elevation':
    1090.0, 'longitude': 32.7934}, u'GE.ISP': {'latitude': 37.8433,
    'local_depth': 5.0, 'elevation': 1100.0, 'longitude': 30.5093},
    u'HL.RDO': {'latitude': 41.146, 'local_depth': 0.0, 'elevation': 100.0,
    'longitude': 25.538}, u'HT.SIGR': {'latitude': 39.2114, 'local_depth':
    0.0, 'elevation': 93.0, 'longitude': 25.8553}, u'HT.ALN': {'latitude':
    40.8957, 'local_depth': 0.0, 'elevation': 110.0, 'longitude': 26.0497},
    u'HL.APE': {'latitude': 37.0689, 'local_depth': 0.0, 'elevation':
    620.0, 'longitude': 25.5306}}
lasif.visualization.plot_stations_for_event(map_object=map,
    station_dict=stations, event_info=event_info)
# Create event.
events = [{
    "filename": "example_2.xml",
    "event_name": "2",
    "latitude": 38.82,
    "longitude": 40.14,
    "depth_in_km": 10,
    "origin_time": UTCDateTime(2013, 1, 1),
    "m_rr": 5.47e+15,
    "m_tt": -4.11e+16,
    "m_pp": 3.56e+16,
    "m_rt": 2.26e+16,
    "m_rp": -2.25e+16,
    "m_tp": 1.92e+16,
    "magnitude": 5.1,
    "magnitude_type": "Mw"}]
lasif.visualization.plot_events(events, map)
plt.show()