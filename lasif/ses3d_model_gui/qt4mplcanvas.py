from matplotlib import rc as matplotlibrc
matplotlibrc('figure.subplot', left=0.0, right=1.0, bottom=0.0, top=1.0)
import matplotlib as mpl
mpl.rcParams['font.size'] = 10
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import \
    FigureCanvasQTAgg as FigureCanvas


class Qt4MplCanvas(FigureCanvas):
    """
    Class to represent the FigureCanvas widget.
    """
    def __init__(self, parent=None):
        # Standard Matplotlib code to generate the plot
        self.fig = Figure()
        # initialize the canvas where the Figure renders into
        super(Qt4MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
