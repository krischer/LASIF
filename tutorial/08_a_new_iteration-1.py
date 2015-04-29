import lasif.visualization
import matplotlib.pylab as plt

from lasif.function_templates import source_time_function
data = source_time_function.source_time_function(2000, 0.3, 1.0 / 100.0,
                                                 1.0 / 40.0, None)
lasif.visualization.plot_tf(data, 0.3, freqmin=1.0 / 100.0,
                            freqmax=1.0 / 40.0)