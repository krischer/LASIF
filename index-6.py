from lasif.tools import Q_discrete
weights = [2.56184936462, 2.50613220548, 0.0648624201145]
relaxation_times = [1.50088990947, 13.3322250004, 22.5140030575]
Q_discrete.plot(weights, relaxation_times, f_min=1.0 / 100.0,
                f_max=1.0 / 8.0)