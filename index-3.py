import lasif.visualization
map = lasif.visualization.plot_domain(34.1, 42.9, 23.1, 42.9, 1.46,
    rotation_axis=[0.0, 0.0, 1.0], rotation_angle_in_degree=0.0,
    show_plot=False, zoom=True)
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
org.latitude = 39.15
org.longitude = 29.1
t.m_rr = -8.07e+17
t.m_tt = 8.92e+17
t.m_pp = -8.5e+16
t.m_rt = 2.8e+16
t.m_rp = -5.3e+16
t.m_tp = -2.17e+17
ev2 = Event()
cat.append(ev2)
org = Origin()
fm = FocalMechanism()
mt = MomentTensor()
t = Tensor()
ev2.origins.append(org)
ev2.focal_mechanisms.append(fm)
fm.moment_tensor = mt
mt.tensor = t
org.latitude = 38.82
org.longitude = 40.14
t.m_rr = 5.47e+15
t.m_tt = -4.11e+16
t.m_pp = 3.56e+16
t.m_rt = 2.26e+16
t.m_rp = -2.25e+16
t.m_tp = 1.92e+16
lasif.visualization.plot_events(cat, map)