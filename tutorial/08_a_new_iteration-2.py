from lasif.tools import Q_discrete
weights = [1.6264684983257656, 1.0142952434286228, 1.5007527644957979]
relaxation_times = [0.68991741458188449, 4.1538611409236301,
                    23.537531778655516]

Q_discrete.plot(weights, relaxation_times, f_min=1.0 / 100.0,
                f_max=1.0 / 10.0)