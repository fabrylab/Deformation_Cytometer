from __future__ import division, print_function
import qtawesome as qta
import sys
from pathlib import Path
import json

""" some magic to prevent PyQt5 from swallowing exceptions """
# Back up the reference to the exceptionhook
sys._excepthook = sys.excepthook
# Set the exception hook to our wrapping function
sys.excepthook = lambda *args: sys._excepthook(*args)

import sys, os, ctypes
from qtpy import QtCore, QtGui, QtWidgets

from qimage2ndarray import array2qimage

import time
import numpy as np
import matplotlib.pyplot as plt
from includes.QExtendedGraphicsView import QExtendedGraphicsView
from includes.MemMap import MemMap
import configparser
import io
import re
import datetime
from pathlib import Path


class QInput(QtWidgets.QWidget):
    """
    A base class for input widgets with a text label and a unified API.

    - The valueChanged signal is emitted when the user has changed the input.

    - The value of the input element get be set with setValue(value) and queried by value()

    """
    # the signal when the user has changed the value
    valueChanged = QtCore.Signal('PyQt_PyObject')

    no_signal = False

    last_emited_value = None

    def __init__(self, layout=None, name=None, tooltip=None, stretch=False):
        # initialize the super widget
        super(QInput, self).__init__()

        # initialize the layout of this widget
        QtWidgets.QHBoxLayout(self)
        self.layout().setContentsMargins(0, 0, 0, 0)

        # add me to a parent layout
        if layout is not None:
            if stretch is True:
                self.wrapper_layout = QtWidgets.QHBoxLayout()
                self.wrapper_layout.setContentsMargins(0, 0, 0, 0)
                self.wrapper_layout.addWidget(self)
                self.wrapper_layout.addStretch()
                layout.addLayout(self.wrapper_layout)
            else:
                layout.addWidget(self)

        # add a label to this layout
        self.label = QtWidgets.QLabel(name)
        self.layout().addWidget(self.label)

        if tooltip is not None:
            self.setToolTip(tooltip)

    def setLabel(self, text):
        # update the label
        self.label.setText(text)

    def _emitSignal(self):
        if self.value() != self.last_emited_value:
            self.valueChanged.emit(self.value())
            self.last_emited_value = self.value()

    def _valueChangedEvent(self, value):
        if self.no_signal:
            return
        self.setValue(value)
        self._emitSignal()

    def setValue(self, value):
        self.no_signal = True
        try:
            self._doSetValue(value)
        finally:
            self.no_signal = False

    def _doSetValue(self, value):
        # dummy method to be overloaded by child classes
        pass

    def value(self):
        # dummy method to be overloaded by child classes
        pass


class QInputNumber(QInput):
    slider_dragged = False

    def __init__(self, layout=None, name=None, value=0, min=None, max=None, use_slider=False, float=True, decimals=2,
                 unit=None, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        if float is False:
            self.decimals = 0
        else:
            if decimals is None:
                decimals = 2
            self.decimals = decimals
        self.decimal_factor = 10**self.decimals

        if use_slider and min is not None and max is not None:
            # slider
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.layout().addWidget(self.slider)
            self.slider.setRange(min * self.decimal_factor, max * self.decimal_factor)
            self.slider.valueChanged.connect(lambda x: self._valueChangedEvent(x / self.decimal_factor))
            self.slider.sliderPressed.connect(lambda: self._setSliderDragged(True))
            self.slider.sliderReleased.connect(lambda: self._setSliderDragged(False))
        else:
            self.slider = None

        # add spin box
        if float:
            self.spin_box = QtWidgets.QDoubleSpinBox()
            self.spin_box.setDecimals(decimals)
        else:
            self.spin_box = QtWidgets.QSpinBox()
        if unit is not None:
            self.spin_box.setSuffix(" " + unit)
        self.layout().addWidget(self.spin_box)
        self.spin_box.valueChanged.connect(self._valueChangedEvent)

        if min is not None:
            self.spin_box.setMinimum(min)
        else:
            self.spin_box.setMinimum(-99999)
        if max is not None:
            self.spin_box.setMaximum(max)
        else:
            self.spin_box.setMaximum(+99999)

        self.setValue(value)

    def _setSliderDragged(self, value):
        self.slider_dragged = value
        if value is False:
            self._emitSignal()

    def _valueChangedEvent(self, value):
        if self.no_signal:
            return
        self.setValue(value)
        if not self.slider_dragged:
            self._emitSignal()

    def _doSetValue(self, value):
        self.spin_box.setValue(value)
        if self.slider is not None:
            self.slider.setValue(value * self.decimal_factor)

    def value(self):
        return self.spin_box.value()


class QInputFilename(QInput):
    last_folder = None

    def __init__(self, layout=None, name=None, value=None, dialog_title="Choose File", file_type="All", filename_checker=None, existing=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        self.dialog_title = dialog_title
        self.file_type = file_type
        self.filename_checker = filename_checker
        self.existing = existing

        self.line = QtWidgets.QLineEdit()
        self.layout().addWidget(self.line)
        self.line.setEnabled(False)

        self.button = QtWidgets.QPushButton("choose file")
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._openDialog)

        # set the color
        self.setValue(value)
        if value is None:
            self.last_folder = os.getcwd()

    def _openDialog(self):
        # open an new files
        if not self.existing:
            filename = QtWidgets.QFileDialog.getSaveFileName(None, self.dialog_title, self.last_folder, self.file_type)
        # or choose an existing file
        else:
            filename = QtWidgets.QFileDialog.getOpenFileName(None, self.dialog_title, self.last_folder, self.file_type)

        # get the string
        if isinstance(filename, tuple):  # Qt5
            filename = filename[0]
        else:  # Qt4
            filename = str(filename)

        # optical check the filename
        if self.filename_checker and filename:
            filename = self.filename_checker(filename)

        # set the filename
        if filename:
            self.setValue(filename)
            self._emitSignal()

    def _doSetValue(self, value):
        self.last_folder = os.path.dirname(value)
        self.line.setText(value)

    def value(self):
        # return the color
        return self.line.text()

class QInputFolder(QInput):
    last_folder = None

    def __init__(self, layout=None, name=None, value=None, dialog_title="Choose File", file_type="All", filename_checker=None, existing=False, **kwargs):
        # initialize the super widget
        QInput.__init__(self, layout, name, **kwargs)

        self.dialog_title = dialog_title
        self.file_type = file_type
        self.filename_checker = filename_checker
        self.existing = existing

        self.line = QtWidgets.QLineEdit()
        self.layout().addWidget(self.line)
        self.line.setEnabled(False)

        self.button = QtWidgets.QPushButton("choose file")
        self.layout().addWidget(self.button)
        self.button.clicked.connect(self._openDialog)

        # set the color
        self.setValue(value)
        if value is None:
            self.last_folder = os.getcwd()

    def _openDialog(self):
        # open an new files
        filename = QtWidgets.QFileDialog.getExistingDirectory(None, self.dialog_title, self.last_folder)

        # get the string
        if isinstance(filename, tuple):  # Qt5
            filename = filename[0]
        else:  # Qt4
            filename = str(filename)

        # optical check the filename
        if self.filename_checker and filename:
            filename = self.filename_checker(filename)

        # set the filename
        if filename:
            self.setValue(filename)
            self._emitSignal()

    def _doSetValue(self, value):
        self.last_folder = value#os.path.dirname(value)
        self.line.setText(value)

    def value(self):
        # return the color
        return self.line.text()

file_settings_last_filename = Path("settings.txt")
file_settings_last_config = Path("last_config.txt")

class QConfigInput(QtWidgets.QWidget):
    signal_start = QtCore.Signal(str, int)
    signal_stop = QtCore.Signal()

    recording = False

    def __init__(self, parent, layout):
        super().__init__()
        layout.addWidget(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        #self.setContentsMargins(0, 0, 0, 0)
        self.setMinimumWidth(300)

        config_cam = configparser.ConfigParser()
        config_cam.read("config.txt")
        self.smap = MemMap(config_cam["camera"]["settings_mmap"])

        file_camera_memmap = "camera.xml"

        while not Path(file_camera_memmap).exists():
            time.sleep(0.01)
            continue

        self.mmap = MemMap(file_camera_memmap)

        self.filename = QInputFolder(layout, "", "config.txt")

        self.text = QtWidgets.QPlainTextEdit()
        self.layout().addWidget(self.text)

        self.button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.button_layout)

        self.button_update = QtWidgets.QPushButton("update")
        self.button_update.clicked.connect(self.update)
        self.button_layout.addWidget(self.button_update)

        self.button_start = QtWidgets.QPushButton("start")
        self.button_start.clicked.connect(self.start)
        self.button_layout.addWidget(self.button_start)

        self.text.setPlainText("""
[Default]
version = 1

[SETUP]
pressure = 20 kPa
channel width = 200 um
channel length = 5.8 cm
imaging position after inlet = 2.9 cm
bioink = alginate 2 pc + rpmi
room temperature = 24 deg C
cell temperature = 24 deg C

[MICROSCOPE]
microscope = Leica DM 6000
objective = 40 x
na = 0.6
coupler = 0.5 x
condensor aperture = 8

[CAMERA]
exposure time = 30 us
gain = 6
frame rate = 500 fps
camera = Basler acA20-520
camera pixel size = 6.9 um
duration = 20 s

[CELL]
cell type = k562
cell passage number = 33
time after harvest = 15 min
treatment = none


        """.strip())

        file = None
        if file_settings_last_filename.exists():
            with file_settings_last_filename.open() as fp:
                try:
                    file = Path(fp.read().strip())
                except ValueError:
                    file = None
        if file is not None:
            self.filename.setValue(str(file))

        if file_settings_last_config.exists():
            with file_settings_last_config.open() as fp:
                self.text.setPlainText(fp.read())
                self.update()


    def update(self):
        config = configparser.ConfigParser()
        t = self.text.toPlainText().strip()#.encode("utf-8")
        config.read_string(t)
        print(config)
        print(config.sections())
        self.mmap.framerate = int(config["CAMERA"]["frame rate"].split()[0])
        t = re.sub(r"gain\s*=\s*(.*)\s*\n", f"gain = {self.mmap.gain}\n", t)
        self.text.setPlainText(t)
        self.config = config
        self.mmap.cells_filled[:] = False

    def start(self):
        if self.recording is True:
            self.button_start.setDisabled(True)
            self.signal_stop.emit()
            return
        self.update()
        fps = int(self.config["CAMERA"]["frame rate"].split()[0])
        duration = float(self.config["CAMERA"]["duration"].split()[0])
        count = fps*duration
        filename = os.path.join(self.filename.value(), datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        with open(filename+"_config.txt", "w") as fp:
            self.config.write(fp)
        with open(file_settings_last_config, "w") as fp:
            self.config.write(fp)

        with file_settings_last_filename.open("w") as fp:
            fp.write(self.filename.value())

        self.signal_start.emit(filename+".tif", int(count))
        print("Record", filename+".tif", int(count))
        self.recording = True
        self.button_start.setText("stop")

        for widget in [self.text, self.button_update, self.filename]:
            widget.setDisabled(True)

    def recoding_stopped(self):
        self.recording = False
        self.button_start.setText("start")
        for widget in [self.text, self.button_update, self.button_start, self.filename]:
            widget.setDisabled(False)
