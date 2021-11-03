from pathlib import Path
import subprocess
import time
import os

from qtpy import QtCore, QtGui, QtWidgets


import time
import numpy as np
import matplotlib.pyplot as plt

from includes.MemMap import MemMap
from qtpy import QtCore, QtGui, QtWidgets
from qimage2ndarray import array2qimage
import numpy as np
from includes.QExtendedGraphicsView import QExtendedGraphicsView
import configparser
from includes.gigecam import GigECam


class QLiveDisplay(QtWidgets.QWidget):
    last_timestamp = 0

    def __init__(self, parent, layout):
        super().__init__()

        self.view = QExtendedGraphicsView()
        self.pixmap = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(1344, 1024), self.view.origin)
        layout.addWidget(self.view)

        self.cam = GigECam()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateview)
        self.timer.start(100)

        file_camera_memmap = "camera.xml"

        while not Path(file_camera_memmap).exists():
            time.sleep(0.01)
            continue

        self.mmap = MemMap(file_camera_memmap)

    def keyPressEvent(self, event):
        # @key ---- General ----

        if event.key() == QtCore.Qt.Key_F:
            # @key F: fit image to view
            self.view.fitInView()

    def updateview(self):
        index = np.argmax(self.mmap.ring_buffer[0].indices)
        time = self.mmap.ring_buffer[0].indices[index]
        if time != self.last_timestamp:
            self.last_timestamp = time
            self.im = self.mmap.ring_buffer[0].images[index].copy()

            # display the iamge
            self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(self.im)))
            self.view.setExtend(self.im.shape[1], self.im.shape[0])
        return