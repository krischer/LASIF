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
org.latitude = 35.9
org.longitude = -10.37
t.m_rr = 6.29e+17
t.m_tt = -1.12e+18
t.m_pp = 4.88e+17
t.m_rt = -2.8e+17
t.m_rp = -5.22e+17
t.m_tp = 3.4e+16
lasif.visualization.plot_events(cat, map)