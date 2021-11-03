
from pathlib import Path
import subprocess
import time
import os

from qtpy import QtCore, QtGui, QtWidgets


import time
import numpy as np
import matplotlib.pyplot as plt

from includes.MemMap import MemMap
from includes.UNETmodel import UNet
from includes.regionprops import save_cells_to_file, mask_to_cells, getTimestamp, getRawVideo, preprocess, matchVelocities, saveResults
from includes.matplotlib_widget import MatplotlibWidget
from includes.gigecam import GigECam
import threading
import pandas as pd
import tifffile


class QLiveAnalysis(QtWidgets.QWidget):
    signal_cells_updated = QtCore.Signal()
    signal_recording_stopped = QtCore.Signal()

    last_cell_update = 0

    recording = False

    def __init__(self, parent, layout):
        super().__init__()
        layout.addWidget(self)
        QtWidgets.QVBoxLayout(self)
        self.setMinimumWidth(600)

        file_camera_memmap = "camera.xml"
        if Path(file_camera_memmap).exists():
            Path(file_camera_memmap).unlink()
        if 1:
            self.process_record = subprocess.Popen(
                ["python", "background_record.py"])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.process_segment = subprocess.Popen(
                ["python", "background_segments.py"])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.process_analyse = subprocess.Popen(
                ["python", "background_analyse.py"])#, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while not Path(file_camera_memmap).exists():
            time.sleep(0.01)
            continue

        self.mmap = MemMap(file_camera_memmap)

        #self.last_cell_update = self.mmap.last_cell_update

        self.progress_bar = QtWidgets.QProgressBar()
        self.layout().addWidget(self.progress_bar)

        self.progress_bar2 = QtWidgets.QProgressBar()
        self.layout().addWidget(self.progress_bar2)

        self.progress_bar3 = QtWidgets.QProgressBar()
        self.layout().addWidget(self.progress_bar3)

        self.canvas = MatplotlibWidget()
        self.layout().addWidget(self.canvas)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateview)
        self.timer.start(100)

        self.filename_cells = Path("cells.txt")
        self.filename_images = Path("images.tif")

    def start_record(self, filename, count):
        if self.filename_cells.exists():
            self.filename_cells.unlink()
        if self.filename_images.exists():
            self.filename_images.unlink()

        self.filename_target = filename

        self.recording = True
        self.record_count = count
        self.progress_bar.setRange(0, count)
        self.progress_bar2.setRange(0, count)
        self.progress_bar3.setRange(0, count)
        self.mmap.record_count = count
        self.mmap.start_recording = True

    def updateview(self):
        if self.mmap.last_cell_update != self.last_cell_update:
            if self.recording:
                self.progress_bar.setValue(self.mmap.images_recorded)
                self.progress_bar2.setValue(self.mmap.images_segmented)
                self.progress_bar3.setValue(self.mmap.images_analysed)
                if not self.mmap.start_recording and not self.mmap.recording:
                    self.signal_recording_stopped.emit()
                    self.filename_images.rename(self.filename_target)
                    self.filename_cells.rename(self.filename_target.replace(".tif", "_cells.txt"))
                    self.recording = False
                    self.signal_recording_stopped.emit()

            im_shape = self.mmap.ring_buffer[0].images[0].shape
            self.last_cell_update = self.mmap.last_cell_update

            cells = self.mmap.cells[self.mmap.cells_filled, :].copy()
            x = cells[:, 1]
            y = cells[:, 2]
            vel = cells[:, -2]
            strain = (cells[:, 4] - cells[:, 5]) / np.sqrt(cells[:, 4] * cells[:, 5])

            plt.clf()
            plt.subplot(221)
            plt.plot(x, y, "o")
            plt.xlabel("x")
            plt.xlim(0, im_shape[1])
            plt.ylabel("y")
            plt.ylim(0, im_shape[0])
            #plt.plot(df["x_pos"], df["y_pos"], "o")
            #rint(df["radial_pos"])
            #print(df["velocity"])
            #plt.xlim(-self.detector.config["channel_width_px"]/2 * self.detector.config["pixel_size_m"] * 1e6, self.detector.config["channel_width_px"]/2 * self.detector.config["pixel_size_m"] * 1e6)
            plt.subplot(222)
            plt.plot(y, vel*1e3, "o")  # from m/s to mm/s
            plt.xlabel("y position (px)")
            plt.xlim(0, im_shape[0])
            plt.ylabel("velocity (mm/s)")
            plt.ylim(bottom=0)
            plt.subplot(223)
            plt.plot(y, strain, "o")
            plt.xlim(0, im_shape[0])
            plt.ylim(bottom=0)
            self.canvas.draw()

    def closeEvent(self, event):
        print("close")
        self.mmap.running = False
        self.process_record.terminate()
        self.process_segment.terminate()
        self.process_analyse.terminate()
        print("processes terminated")

if __name__ == "__main__":
    import subprocess
    import time
    import os
    os.chdir("..")
    print("start process")

    mmap.running = False

    time.sleep(10)