import lasif.visualization
import matplotlib.pylab as plt

from lasif.source_time_functions import filtered_heaviside
data = filtered_heaviside(2000, 0.3, 1.0 / 100.0, 1.0 / 40.0)
lasif.visualization.plot_tf(data, 0.3, freqmin=1.0 / 100.0,
                            freqmax=1.0 / 40.0)