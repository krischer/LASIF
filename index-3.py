import lasif.visualization
map = lasif.visualization.plot_domain(-20, +20, -20, +20, 3.0,
    rotation_axis=[1.0, 1.0, 1.0], rotation_angle_in_degree=-45.0,
    show_plot=False)
# Create event.
from obspy.core.event import *
ev = Event()
cat = Catalog(events=[ev])
org = Origin()
fm = FocalMechanism()
mt = MomentTensor()
t = Tensor()
ev.origins.append(org)
ev.focal_mechanisms.append(fm)
fm.moment_tensor = mt
mt.tensor = t
org.latitude = 37.4
org.longitude = -24.38
t.m_rr = -1.69e+18
t.m_tt = 9.12e+17
t.m_pp = 7.77e+17
t.m_rt = 8.4e+16
t.m_rp = 2.4e+16
t.m_tp = -4.73e+17
lasif.visualization.plot_events(cat, map)
ev_lng = -24.38
ev_lat = 37.4
stations = {'GE.SFS': {'latitude': 36.4656, 'local_depth': 5.0,
    'elevation': 21.0, 'longitude': -6.2055}, 'PM.MTE': {'latitude':
    40.3997, 'local_depth': 3.0, 'elevation': 815.0, 'longitude': -7.5442},
    'PM.PVAQ': {'latitude': 37.4037, 'local_depth': 0.0, 'elevation':
    200.0, 'longitude': -7.7173}, 'WM.CART': {'latitude': 37.5868,
    'local_depth': 5.0, 'elevation': 65.0, 'longitude': -1.0012}, 'GE.MTE':
    {'latitude': 40.3997, 'local_depth': 3.0, 'elevation': 815.0,
    'longitude': -7.5442}, 'PM.PESTR': {'latitude': 38.8672, 'local_depth':
    0.0, 'elevation': 410.0, 'longitude': -7.5902}, 'GE.CART': {'latitude':
    37.5868, 'local_depth': 5.0, 'elevation': 65.0, 'longitude': -1.0012},
    'IU.PAB': {'latitude': 39.5446, 'local_depth': 0.0, 'elevation': 950.0,
    'longitude': -4.349899}}
lasif.visualization.plot_stations_for_event(map_object=map,
    station_dict=stations, event_longitude=ev_lng,
    event_latitude=ev_lat)
# Plot the beachball for one event.
lasif.visualization.plot_events(cat, map_object=map)