import sys

# Setting the Qt bindings for QtPy
import os
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
from qtpy import API_NAME as QT_API_NAME
if QT_API_NAME.startswith("PyQt4"):
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt4agg import FigureManager
    from matplotlib.backends.backend_qt4 import NavigationToolbar2QT as NavigationToolbar
else:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
    from matplotlib.backends.backend_qt5agg import FigureManager
    from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import _pylab_helpers
from pathlib import Path
import numpy as np
import imageio
from qimage2ndarray import array2qimage
import matplotlib.pyplot as plt
import yaml

from deformationcytometer.evaluation.helper_functions import getMeta, load_all_data_new, plot_velocity_fit, plotDensityScatter, plot_density_hist, plotBinnedData

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)


def QUrl2PythonPath(url):
    url = str(url.toString()).strip()
    if url.startswith("file:///"):
        url = url[len("file:///"):]
    if url.startswith("file:"):
        url = url[len("file:"):]
    return Path(url)

class MatplotlibWidget(Canvas):

    def __init__(self, parent=None, width=4, height=3, dpi=100):
        plt.ioff()
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.patch.set_facecolor([0, 1, 0, 0])
        self.axes = self.figure.add_subplot(111)

        Canvas.__init__(self, self.figure)
        self.setParent(parent)

        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.manager = FigureManager(self, 1)
        self.manager._cidgcf = self.figure

        """
        _pylab_helpers.Gcf.figs[num] = canvas.manager
        # get the canvas of the figure
        manager = _pylab_helpers.Gcf.figs[num]
        # set the size if it is defined
        if figsize is not None:
            _pylab_helpers.Gcf.figs[num].window.setGeometry(100, 100, figsize[0] * 80, figsize[1] * 80)
        # set the figure as the active figure
        _pylab_helpers.Gcf.set_active(manager)
        """
        _pylab_helpers.Gcf.set_active(self.manager)

def pathParts(path):
    if path.parent == path:
        return [path]
    return pathParts(path.parent) + [path]


class MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        try:
            self.setWindowIcon(qta.icon("mdi.folder-pound-outline"))
        except Exception:
            pass

        # QSettings
        self.settings = QtCore.QSettings("DeformationCytometer", "DeformationCytometer")

        self.setMinimumWidth(1200)
        self.setMinimumHeight(400)
        self.setWindowTitle("DeformationCytometer Viewer")

        hlayout = QtWidgets.QHBoxLayout(self)

        self.browser = Browser()
        #hlayout.addWidget(self.browser)

        self.plot = MeasurementPlot()
        #hlayout.addWidget(self.plot)

        self.text = MetaDataEditor()
        #hlayout.addWidget(self.text)

        self.splitter_filebrowser = QtWidgets.QSplitter()
        self.splitter_filebrowser.addWidget(self.browser)
        self.splitter_filebrowser.addWidget(self.plot)
        self.splitter_filebrowser.addWidget(self.text)
        hlayout.addWidget(self.splitter_filebrowser)
        #splitter_filebrowser.setStretchFactor(0, 2)
        #splitter_filebrowser.setStretchFactor(1, 4)

        self.browser.signal_selection_changed.connect(self.selected)

    def selected(self, name):
        self.text.selected(name)
        self.plot.selected(name)


class MeasurementPlot(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.hlayout = QtWidgets.QVBoxLayout(self)
        self.hlayout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MatplotlibWidget(self)
        plt.clf()
        self.hlayout.addWidget(self.canvas)
        self.tools = NavigationToolbar(self.canvas, self)
        self.hlayout.addWidget(self.tools)

    def selected(self, name):
        plt.clf()
        if name.endswith(".tif"):
            data, config = load_all_data_new(name.replace(".tif", "_evaluated_new.csv"), do_excude=False)

            plt.subplot(3, 3, 1)
            plot_velocity_fit(data)
            if "vel_fit_error" in data.iloc[0]:
                plt.text(0.8, 0.8, f'{data.iloc[0]["vel_fit_error"]:.0f}', transform=plt.gca().transAxes)

            plt.subplot(3, 3, 2)
            plt.axline([0,0], slope=1, color="k")
            plt.plot(data.omega, data.omega_weissenberg, "o", ms=1)
            plt.xlabel("omega")
            plt.ylabel("omega weissenberg")

            plt.subplot(3, 3, 3)
            plotDensityScatter(data.stress, data.epsilon)
            plotBinnedData(data.stress, data.epsilon, bins=np.arange(0, 300, 10))
            plt.xlabel("stress (Pa)")
            plt.ylabel("strain")

            plt.subplot(3, 3, 4)
            plt.loglog(data.omega_weissenberg, data.w_Gp1, "o", alpha=0.25, ms=1)
            plt.loglog(data.omega_weissenberg, data.w_Gp2, "o", alpha=0.25, ms=1)
            from scipy.special import gamma
            def fit(omega, k, alpha):
                omega = np.array(omega)
                G = k * (1j * omega) ** alpha * gamma(1 - alpha)
                return np.real(G), np.imag(G)

            def cost(p):
                Gp1, Gp2 = fit(data.omega_weissenberg, *p)
                return np.sum((np.log10(data.w_Gp1) - np.log10(Gp1)) ** 2) + np.sum(
                    (np.log10(data.w_Gp2) - np.log10(Gp2)) ** 2)

            from scipy.optimize import minimize
            res = minimize(cost, [np.median(data.w_k_cell), np.mean(data.w_alpha_cell)], bounds=([0, np.inf], [0, 1]))
            print(res)
            xx = [10**np.floor(np.log10(np.min(data.w_Gp1))), 10**np.ceil(np.log10(np.max(data.w_Gp1)))]
            plt.plot(xx, fit(xx, *res.x)[0], "k-", lw=0.8)
            plt.plot(xx, fit(xx, *res.x)[1], "k--", lw=0.8)

            plt.plot(xx, fit(xx, np.median(data.w_k_cell), np.mean(data.w_alpha_cell))[0], "r-", lw=0.8)
            plt.plot(xx, fit(xx, np.median(data.w_k_cell), np.mean(data.w_alpha_cell))[1], "r--", lw=0.8)

            plt.ylabel("G' / G''")
            plt.xlabel("angular frequency")

            plt.subplot(3, 3, 5)
            plt.cla()
            plt.xlim(0, 4)
            plot_density_hist(np.log10(data.w_k_cell), color="C0")
            plt.xlabel("log10(k)")
            plt.ylabel("relative density")
            plt.text(0.9, 0.9,
                     f"mean(log10(k)) {np.mean(np.log10(data.w_k_cell)):.2f}\nstd(log10(k)) {np.std(np.log10(data.w_k_cell)):.2f}\nmean(k) {np.mean(data.k_cell):.2f}\nstd(k) {np.std(data.k_cell):.2f}\n",
                     transform=plt.gca().transAxes, va="top", ha="right")

            plt.subplot(3, 3, 6)
            plt.cla()
            plt.xlim(0, 1)
            plot_density_hist(data.w_alpha_cell, color="C1")
            plt.xlabel("alpha")
            plt.text(0.9, 0.9,
                     f"mean($\\alpha$) {np.mean(data.w_alpha_cell):.2f}\nstd($\\alpha$) {np.std(data.w_alpha_cell):.2f}\n",
                     transform=plt.gca().transAxes, va="top", ha="right")

            plt.tight_layout()
            #plt.plot(data.rp, data.vel)
        self.canvas.draw()


class MetaDataEditor(QtWidgets.QWidget):
    yaml_file = None

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        hlayout = QtWidgets.QVBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 0, 0)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setToolTip("Meta data from parent folders")
        hlayout.addWidget(self.text)

        self.text2 = QtWidgets.QPlainTextEdit()
        self.text2.textChanged.connect(self.save)
        self.text2.setToolTip("Meta data from current folder/file. Can be editied and will be automatically saved")
        hlayout.addWidget(self.text2)

        self.name = QtWidgets.QLineEdit()
        self.name.setReadOnly(True)
        self.name.setToolTip("The current folder/file.")
        hlayout.addWidget(self.name)

    def save(self):
        if self.yaml_file is not None:
            with open(self.yaml_file, "w") as fp:
                fp.write(self.text2.toPlainText())

    def selected(self, name):
        meta = getMeta(name)
        self.name.setText(name)

        self.text.setPlainText(yaml.dump(meta))

        self.yaml_file = None
        if name.endswith(".tif"):
            yaml_file = Path(name.replace(".tif", "_meta.yaml"))
        else:
            yaml_file = Path(name) / "meta.yaml"

        if yaml_file.exists():
            with yaml_file.open() as fp:
                self.text2.setPlainText(fp.read())
        else:
            self.text2.setPlainText("")
        self.yaml_file = yaml_file


class Browser(QtWidgets.QTreeView):
    signal_selection_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.settings = QtCore.QSettings("fabrylab", "flowcytometer browser")

        # self.setCentralWidget(self.frame)
        #hlayout = QtWidgets.QVBoxLayout(self)

        """ browser"""
        self.dirmodel = QtWidgets.QFileSystemModel()
        # Don't show files, just folders
        # self.dirmodel.setFilter(QtCore.QDir.NoDotAndDotDot | QtCore.QDir.AllDirs)
        self.dirmodel.setNameFilters(["*.tif"])
        self.dirmodel.setNameFilterDisables(False)
        self.folder_view = self#QtWidgets.QTreeView(parent=self)
        self.folder_view.setModel(self.dirmodel)
        self.folder_view.activated[QtCore.QModelIndex].connect(self.clicked)
        # self.folder_view.selected[QtCore.QModelIndex].connect(self.clicked)

        # Don't show columns for size, file type, and last modified
        self.folder_view.setHeaderHidden(True)
        self.folder_view.hideColumn(1)
        self.folder_view.hideColumn(2)
        self.folder_view.hideColumn(3)

        self.selectionModel = self.folder_view.selectionModel()

        #hlayout.addWidget(self.folder_view)

        if self.settings.value("browser/path"):
            self.set_path(self.settings.value("browser/path"))
        else:
            self.set_path(r"\\131.188.117.96\biophysDS")

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        # accept url lists (files by drag and drop)
        for url in event.mimeData().urls():
            url = QUrl2PythonPath(url)
            if url.is_dir() or url.suffix == ".tif":
                event.accept()
                return
        event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent):
        event.acceptProposedAction()

    def dropEvent(self, event: QtCore.QEvent):
        for url in event.mimeData().urls():
            url = QUrl2PythonPath(url)
            if url.is_dir() or url.suffix == ".tif":
                self.set_path(url)

    def set_path(self, path):
        path = Path(path)
        self.dirmodel.setRootPath(str(path.parent))
        for p in pathParts(path):
            self.folder_view.expand(self.dirmodel.index(str(p)))
        self.folder_view.setCurrentIndex(self.dirmodel.index(str(path)))
        print("scroll to ", str(path), self.dirmodel.index(str(path)))
        self.folder_view.scrollTo(self.dirmodel.index(str(path)))

    def clicked(self, index):
        # get selected path of folder_view
        index = self.selectionModel.currentIndex()
        dir_path = self.dirmodel.filePath(index)
        print(dir_path)
        self.settings.setValue("browser/path", dir_path)
        self.signal_selection_changed.emit(dir_path)

        if dir_path.endswith(".npz"):
            print("################# load", dir_path)
            self.loadFile(dir_path)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # set an application id, so that windows properly stacks them in the task bar
    if sys.platform[:3] == 'win':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('fabrybiophysics.deformationcytometer_browser')  # arbitrary string
    print(sys.argv)
    window = MainWindow()
    if len(sys.argv) >= 2:
        window.loadFile(sys.argv[1])
    window.show()
    sys.exit(app.exec_())
