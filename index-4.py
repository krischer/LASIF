import lasif.visualization
import obspy
import numpy as np
def filtered_heaviside(npts, delta, freqmin, freqmax):
    trace = obspy.Trace(data=np.ones(npts))
    trace.stats.delta = delta
    trace.filter("lowpass", freq=freqmax, corners=5)
    trace.filter("highpass", freq=freqmin, corners=2)
    return trace.data
data = filtered_heaviside(1500, 0.75, 1.0 / 500.0, 1.0 / 60.0)
lasif.visualization.plot_tf(data, 0.75)