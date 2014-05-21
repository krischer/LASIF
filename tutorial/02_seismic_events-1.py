import lasif.visualization
from obspy import UTCDateTime
bmap = lasif.visualization.plot_domain(-10.0, 10.0, -10.0, 10.0, 2.5,
    rotation_axis=[1.0, 1.0, 0.2], rotation_angle_in_degree=-65.0,
    plot_simulation_domain=False, zoom=True)
events = [{
    'depth_in_km': 9.0,
    'event_name': 'GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20-2',
    'filename': 'GCMT_event_NORTHWESTERN_BALKAN_REGION_Mag_5.9_1980-5-18-20-2.xml',
    'latitude': 43.29,
    'longitude': 20.84,
    'm_pp': -4.449e+17,
    'm_rp': -5.705e+17,
    'm_rr': -1.864e+17,
    'm_rt': 2.52e+16,
    'm_tp': 4.049e+17,
    'm_tt': 6.313e+17,
    'magnitude': 5.9,
    'magnitude_type': 'Mwc',
    'origin_time': UTCDateTime(1980, 5, 18, 20, 2, 57, 500000),
    'region': u'NORTHWESTERN BALKAN REGION'
}, {
    'depth_in_km': 10.0,
    'event_name': 'GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17-14',
    'filename': 'GCMT_event_NORTHERN_ITALY_Mag_4.9_2000-8-21-17-14.xml',
    'latitude': 44.87,
    'longitude': 8.48,
    'm_pp': 1.189e+16,
    'm_rp': -1600000000000000.0,
    'm_rr': -2.271e+16,
    'm_rt': -100000000000000.0,
    'm_tp': -2.075e+16,
    'm_tt': 1.082e+16,
    'magnitude': 4.9,
    'magnitude_type': 'Mwc',
    'origin_time': UTCDateTime(2000, 8, 21, 17, 14, 27),
    'region': u'NORTHERN ITALY'}]
lasif.visualization.plot_events(events, bmap)