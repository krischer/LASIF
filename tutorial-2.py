import lasif.visualization
from obspy import UTCDateTime
bmap = lasif.visualization.plot_domain(-20.0, 20.0, -20.0, 20.0, 3.0,
    rotation_axis=[1.0, 1.0, 0.2], rotation_angle_in_degree=-65.0,
    plot_simulation_domain=False, zoom=True)
events = [{
    'depth_in_km': 10.0,
    'event_name': 'GCMT_event_PYRENEES_Mag_5.1_1980-2-29-20-40',
    'filename': 'GCMT_event_PYRENEES_Mag_5.1_1980-2-29-20-40.xml',
    'latitude': 43.26,
    'longitude': -0.34,
    'm_pp': -7730000000000000.0,
    'm_rp': -2.617e+16,
    'm_rr': -3.713e+16,
    'm_rt': 3.348e+16,
    'm_tp': -2.662e+16,
    'm_tt': 4.487e+16,
    'magnitude': 5.1,
    'magnitude_type': 'Mwc',
    'origin_time': UTCDateTime(1980, 2, 29, 20, 40, 48, 500000),
    'region': u'PYRENEES'
}, {
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
    'region': u'NORTHWESTERN BALKAN REGION'}]
lasif.visualization.plot_events(events, bmap)