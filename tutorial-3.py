import lasif.visualization
from obspy import UTCDateTime
map = lasif.visualization.plot_domain(34.1, 42.9, 23.1, 42.9, 1.46,
    rotation_axis=[0.0, 0.0, 1.0], rotation_angle_in_degree=0.0,
    zoom=True)
# Create event.
events = [{
    "filename": "example_1.xml",
    "event_name": "1",
    "latitude": 39.15,
    "longitude": 29.1,
    "depth_in_km": 10,
    "origin_time": UTCDateTime(2012, 1, 1),
    "m_rr": -8.07e+17,
    "m_tt": 8.92e+17,
    "m_pp": -8.5e+16,
    "m_rt": 2.8e+16,
    "m_rp": -5.3e+16,
    "m_tp": -2.17e+17,
    "magnitude": 5.1,
    "magnitude_type": "Mw"
}, {
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