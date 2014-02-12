import lasif.visualization

from lasif.source_time_functions import filtered_heaviside

data = filtered_heaviside(4000, 0.13, 1.0 / 500.0, 1.0 / 60.0)
lasif.visualization.plot_tf(data, 0.13)