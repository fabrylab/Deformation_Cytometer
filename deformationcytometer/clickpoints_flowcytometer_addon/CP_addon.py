#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CellDetector.py

# Copyright (c) 2015-2016, Richard Gerum, Sebastian Richter
#
# This file is part of ClickPoints.
#
# ClickPoints is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ClickPoints is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranfty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ClickPoints. If not, see <http://www.gnu.org/licenses/>
import json
import numpy as np
import qtawesome as qta
from qtpy import QtCore, QtWidgets, QtGui
import imageio
import shutil
import clickpoints
from clickpoints.includes.QtShortCuts import AddQSpinBox, AddQOpenFileChoose
from clickpoints.includes import QtShortCuts
import peewee
from inspect import getdoc
import traceback
import os
import sys
import asyncio
from importlib import import_module, reload
import configparser
from skimage.measure import label, regionprops
from deformationcytometer.detection.includes.UNETmodel import UNet
from deformationcytometer.detection.includes.regionprops import mask_to_cells_edge
from pathlib import Path
import pandas as pd
import numpy as np
from clickpoints.includes.matplotlibwidget import MatplotlibWidget, NavigationToolbar
from matplotlib import pyplot as plt
import time
from pathlib import Path
print(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deformationcytometer.includes.includes import getConfig
from deformationcytometer.includes.includes import getInputFile, getConfig, getData
from deformationcytometer.evaluation.helper_functions import getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness, apply_velocity_fit
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plot_density_hist
from deformationcytometer.detection.includes.UNETmodel import store_path
import deformationcytometer.detection.includes.UNETmodel
from deformationcytometer.detection.includes.regionprops import getTimestamp, save_cells_to_file
from PyQt5.QtCore import pyqtSignal
from functools import partial
from deformationcytometer.includes.includes import Dialog
from scipy.stats import gaussian_kde
from PIL import Image
from deformationcytometer.tanktreading.helpers import getCroppedImages, doTracking, CachedImageReader

def doNothing(x):
    return x

default_config_path = Path(deformationcytometer.detection.includes.UNETmodel.__file__).parent
default_config_path = default_config_path.joinpath("default_config.txt")

def X(datafile):
    datafile = datafile.replace(".tif", "_result.txt")
    # %% import raw data
    data = np.genfromtxt(datafile, dtype=float, skip_header=2)
    return data


def stressfunc(R, filename_config):  # imputs (radial position and pressure)
    config = configparser.ConfigParser()
    config.read(filename_config)

    pressure = float(config['SETUP']['pressure'].split()[0]) * 1000  # applied pressure (in Pa)
    channel_width = float(config['SETUP']['channel width'].split()[0]) * 1e-6  # in m
    # channel_width=196*1e-6 #in m
    channel_length = float(config['SETUP']['channel length'].split()[0]) * 1e-2  # in m
    framerate = float(config['CAMERA']['frame rate'].split()[0])  # in m

    magnification = float(config['MICROSCOPE']['objective'].split()[0])
    coupler = float(config['MICROSCOPE']['coupler'].split()[0])
    camera_pixel_size = float(config['CAMERA']['camera pixel size'].split()[0])

    pixel_size = camera_pixel_size / (magnification * coupler)  # in micrometer

    # %% stress profile in channel
    L = channel_length  # length of the microchannel in meter
    H = channel_width  # height(and width) of the channel

    P = -pressure

    G = P / L  # pressure gradient
    pre_factor = (4 * (H ** 2) * G) / (np.pi) ** 3
    u_primy = np.zeros(len(R))
    sumi = 0
    for i in range(0, len(R)):
        for n in range(1, 100, 2):  # sigma only over odd numbers
            u_primey = pre_factor * ((-1) ** ((n - 1) / 2)) * (np.pi / ((n ** 2) * H)) \
                       * (np.sinh((n * np.pi * R[i]) / H) / np.cosh(n * np.pi / 2))
            sumi = u_primey + sumi
        u_primy[i] = sumi
        sumi = 0
    stress = np.sqrt((u_primy) ** 2)
    return stress  # output atress profile


def strain(longaxis, shortaxis):
    D = np.sqrt(longaxis * shortaxis)  # diameter of undeformed (circular) cell
    strain = (longaxis - shortaxis) / D
    return strain

# todo import from defocytometer





class Worker(QtCore.QThread):

    def __init__(self, parent=None, run_function=None):
        QtCore.QThread.__init__(self, parent)
        self.run_function = run_function

    def run(self):
        self.run_function()



class SetFile(QtWidgets.QHBoxLayout):

    fileSeleted = pyqtSignal(bool)
    def __init__(self, file=None, type="file", filetype=""):
        super().__init__() # activating QVboxLayout
        if file is None:
            self.file = ""
        else:
            self.file = file
        self.filetype = filetype
        self.type = type
        #self.folder = os.getcwd()
        # line edit holding the currently selected folder 1
        self.line_edit_folder = QtWidgets.QLineEdit(str(self.file))
        self.line_edit_folder.editingFinished.connect(self.emitTextChanged)
        self.addWidget(self.line_edit_folder, stretch=4)

        # button to browse folders
        self.open_folder_button = QtWidgets.QPushButton("choose files")
        self.open_folder_button.clicked.connect(self.file_dialog)
        self.addWidget(self.open_folder_button, stretch=2)


    def file_dialog(self):
        dialog = Dialog(title="open file", filetype=self.filetype, mode="file", settings_name="Deformationcytometer Addon")
        self.file = dialog.openFile()
        self.fileSeleted.emit(True)
        self.line_edit_folder.setText(self.file)
        # TOD check if that works
    def emitTextChanged(self):
        self.fileSeleted.emit(True)



def plotDensityScatter(x, y, cmap='viridis', alpha=1, skip=1, y_factor=1, s=5, levels=None, loglog=False, ax=None):
    ax = ax if not ax is None else plt.gca()
    x = np.array(x)[::skip]
    y = np.array(y)[::skip]
    filter = ~np.isnan(x) & ~np.isnan(y)
    if loglog is True:
        filter &= (x>0) & (y>0)
    x = x[filter]
    y = y[filter]
    if loglog is True:
        xy = np.vstack([np.log10(x), np.log10(y)])
    else:
        xy = np.vstack([x, y*y_factor])
    kde = gaussian_kde(xy)
    kd = kde(xy)
    idx = kd.argsort()
    x, y, z = x[idx], y[idx], kd[idx]
    ax.scatter(x, y, c=z, s=s, alpha=alpha, cmap=cmap)  # plot in kernel density colors e.g. viridis

    if levels != None:
        X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
        print(xy.shape)
        print(np.dstack([X, Y*y_factor]).shape)
        XY = np.dstack([X, Y*y_factor])
        Z = kde(XY.reshape(-1, 2).T).reshape(XY.shape[:2])
        ax.contour(X, Y, Z, levels=1)

    if loglog is True:
        ax.loglog()

data_keys = ['frames', 'x', 'y', 'rp', 'long_axis', 'short_axis', 'angle',
       'irregularity', 'solidity', 'sharpness', 'timestamp', 'velocity',
       'velocity_partner', 'cell_id']





class Addon(clickpoints.Addon):
    data = None
    data2 = None
    unet = None

    signal_update_plot = QtCore.Signal()
    signal_plot_finished = QtCore.Signal()
    image_plot = None
    last_update = 0
    updating = False
    exporting = False
    exporting_index = 0

    def __init__(self, *args, **kwargs):
        clickpoints.Addon.__init__(self, *args, **kwargs)

        self.thread = Worker(run_function=None)
        self.unet = None
        self.layout = QtWidgets.QVBoxLayout(self)


        # Check if the marker type is present
        self.marker_type_cell1 = self.db.setMarkerType("cell", "#0a2eff", self.db.TYPE_Ellipse)
        self.marker_type_cell2 = self.db.setMarkerType("cell2", "#Fa2eff", self.db.TYPE_Ellipse)
        self.cp.reloadTypes()

        self.prob_folder = os.environ["CLICKPOINTS_TMP"]
        self.prob_path = self.db.setPath(self.prob_folder)
        self.prob_layer = self.db.setLayer("prob_map")

        clickpoints.Addon.__init__(self, *args, **kwargs)

        # set the title and layout
        self.setWindowTitle("DeformationCytometer - ClickPoints")
        self.layout = QtWidgets.QVBoxLayout(self)
        # weight file selection
        self.weight_selection = SetFile(store_path, filetype="weight file (*.h5)")
        self.weight_selection.fileSeleted.connect(self.initUnet)
        self.layout.addLayout(self.weight_selection)
        # update segmentation
        seg_layout = QtWidgets.QHBoxLayout()
        self.update_detection_button = QtWidgets.QPushButton("update_detection")
        self.update_detection_button.clicked.connect(partial(self.start_threaded, self.detect_all))
        seg_layout.addWidget(self.update_detection_button)
        self.layout.addLayout(seg_layout)
      #  self.segmentation_tickbox = QtWidgets.QCheckBox("update segmentation automatically")
        #seg_layout.addWidget(self.segmentation_tickbox)

        validator = QtGui.QDoubleValidator(0, 100, 3)
        filter_layout = QtWidgets.QHBoxLayout()
        reg_label = QtWidgets.QLabel("irregularity")
        filter_layout.addWidget(reg_label)
        self.reg_box = QtWidgets.QLineEdit("1.06")
        self.reg_box.setValidator(validator)
        filter_layout.addWidget(self.reg_box, stretch=1) # TODO implement text edited method
        sol_label = QtWidgets.QLabel("solidity")
        filter_layout.addWidget(sol_label)
        self.sol_box = QtWidgets.QLineEdit("0.96")
        self.sol_box.setValidator(validator)
        filter_layout.addWidget(self.sol_box, stretch=1)
        filter_layout.addStretch(stretch=5)
        self.layout.addLayout(filter_layout)




        layout = QtWidgets.QHBoxLayout()
        self.button_stressstrain = QtWidgets.QPushButton("stress-strain")
        self.button_stressstrain.clicked.connect(self.plot_stress_strain)
        layout.addWidget(self.button_stressstrain)
        self.button_reg_sol = QtWidgets.QPushButton("reg-sol")
        self.button_reg_sol.clicked.connect(self.plot_irreg)
        layout.addWidget(self.button_reg_sol)

        self.button_kHist = QtWidgets.QPushButton("k_hist")
        self.button_kHist.clicked.connect(self.plot_kHist)
        layout.addWidget(self.button_kHist)

        self.button_alphaHist = QtWidgets.QPushButton("alpha_hist")
        self.button_alphaHist.clicked.connect(self.plot_alphaHist)
        layout.addWidget(self.button_alphaHist)

        self.button_kalpha = QtWidgets.QPushButton("k-alpha")
        self.button_kalpha.clicked.connect(self.plot_k_alpha)
        layout.addWidget(self.button_kalpha)

        # horizontal seperating line
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.VLine)
        frame.setLineWidth(3)
        layout.addWidget(frame)
        self.switch_data_button = QtWidgets.QPushButton("display existing data")
        self.switch_data_button.clicked.connect(self.switch_display_data)
        layout.addWidget(self.switch_data_button)
        self.layout.addLayout(layout)

        # TODO do we really need that
      #  self.button_stressy = QtWidgets.QPushButton("y-strain")
       # self.button_stressy.clicked.connect(self.plot_y_strain)
       # layout.addWidget(self.button_stressy)
        # TODO do we really need that
        #self.button_y_angle = QtWidgets.QPushButton("y-angle")
        #self.button_y_angle.clicked.connect(self.plot_y_angle)
       # layout.addWidget(self.button_y_angle)

        self.layout.addLayout(layout)

        # add a plot widget
        self.plot = MatplotlibWidget(self)
        self.layout.addWidget(self.plot)
        self.layout.addWidget(NavigationToolbar(self.plot, self))
        self.plot.figure.canvas.mpl_connect('button_press_event', self.button_press_callback)

        # add a progress bar
        self.progressbar = QtWidgets.QProgressBar()
        self.layout.addWidget(self.progressbar)

        pbar_info_layout = QtWidgets.QHBoxLayout()
        self.pbarLable = QtWidgets.QLabel("")
        pbar_info_layout.addWidget(self.pbarLable, stretch=1)
        pbar_info_layout.addStretch(stretch=2)
        self.stop_button = QtWidgets.QPushButton("stop")
        self.stop_button.clicked.connect(self.quit_thread)
        pbar_info_layout.addWidget(self.stop_button, stretch=1)
        self.layout.addLayout(pbar_info_layout)

        # connect slots
        # self.signal_update_plot.connect(self.updatePlotImageEvent)
        # self.signal_plot_finished.connect(self.plotFinishedEvent)

        # initialize the table
        # self.updateTable()
        # self.selected = None


        self.filename = self.db.getImage(0).get_full_filename()
        if not os.path.isabs(self.filename): # check dis on windows
            self.filename = str(Path(self.db._database_filename).parent.joinpath(Path(self.filename))) # might not always work
        self.config_file = Path(self.filename.replace(".tif", "_config.txt"))
        self.result_file = Path(self.filename.replace(".tif", "_result.txt"))
        self.addon_result_file = Path(self.filename.replace(".tif", "_addon_result.txt"))
        self.addon_evaluated_file = Path(self.filename.replace(".tif", "_addon_evaluated.csv"))
        self.addon_config_file = Path(self.filename.replace(".tif", "_addon_config.txt"))
        self.vidcap = imageio.get_reader(self.filename)


        self.data_all_existing = pd.DataFrame()
        self.data_mean_existing = pd.DataFrame()
        self.data_all_new = pd.DataFrame()
        self.data_mean_new = pd.DataFrame()


        if self.config_file.exists() and self.result_file.exists():
            self.config = getConfig(self.config_file)
            if "irregularity" in self.config.keys() and "solidity" in self.config.keys():
                solidity_threshold = self.config["solidity"]
                irregularity_threshold = self.config["irregularity"]
            else:
                solidity_threshold = self.sol_threshold
                irregularity_threshold = self.reg_threshold

            self.data_all_existing, self.data_mean_existing = self.load_data(self.result_file,
                                                        solidity_threshold, irregularity_threshold)
            # --> self.data contains stresses and possibly tank treading and weisenberg --> use this for plotting

        else:
            self.config = getConfig(default_config_path) # TODO fix this// save a default config somewhere

        if not self.addon_config_file.exists():
            shutil.copy(self.config_file, self.addon_config_file)

        print("loading finished")
        self.db.deleteEllipses(type=self.marker_type_cell1)
        self.db.deleteEllipses(type=self.marker_type_cell2)
        self.start_threaded(partial(self.display_ellipses, type=self.marker_type_cell1, data=self.data_all_existing))
        # initialize plot
        self.plot_stress_strain()

    @property
    def data_all(self):
        if self.switch_data_button.text() == "display existing data":
            return self.data_all_existing
        if self.switch_data_button.text() == "display new data":
            return self.data_all_new
    @property
    def data_mean(self):
        if self.switch_data_button.text() == "display existing data":
            return self.data_mean_existing
        if self.switch_data_button.text() == "display new data":
            return self.data_mean_new
    @property
    def sol_threshold(self):
        return float(self.sol_box.text())
    @property
    def reg_threshold(self):
        return float(self.reg_box.text())

    def switch_display_data(self):
        if self.switch_data_button.text() == "display existing data":
            self.switch_data_button.setText("display new data")
            return
        if self.switch_data_button.text() == "display new data":
            self.switch_data_button.setText("display existing data")
            return

    def load_data(self, file, solidity_threshold, irregularity_threshold):

        data_all = getData(file)
        if len(data_all) == 0:
            print("no data loaded from file '%s'" % (file))
            return pd.DataFrame(), pd.DataFrame()
        # get velocity generates cell ids
        # TODO use speed and stuff in this step for display soewhere...
        # getVelocity(data_all, self.config)  # todo improve speed in this step // or look for evaluated file
        #data_all = data_all.reindex()
        # TODO: test on windows
        data_mean, config_eval = load_all_data(str(file), solidity_threshold=solidity_threshold,
                                                    irregularity_threshold=irregularity_threshold)  # use a "read sol from config falg here
        return data_all, data_mean



    def button_press_callback(self, event):
        # only drag with left mouse button
        if event.button != 1:
            return
        # if the user doesn't have clicked on an axis do nothing
        if event.inaxes is None:
            return
        # get the pixel of the kymograph
        xy = np.array([event.xdata, event.ydata])
        scale = np.mean(self.plot_data, axis=1)
        distance = np.linalg.norm(self.plot_data / scale[:, None] - xy[:, None] / scale[:, None], axis=0)
        print(self.plot_data.shape, xy[:, None].shape, distance.shape)
        nearest_dist = np.min(distance)
        print("distance ", nearest_dist)
        nearest_point = np.argmin(distance)
        print("clicked", xy)
        self.cp.jumpToFrame(self.data_all.frames[nearest_point])
        self.cp.centerOn(self.data_all.x[nearest_point], self.data_all.y[nearest_point])
        #
        #self.detect_single()


    def plot_alphaHist(self):
        self.plot.axes.clear()
        try:
            x = self.data_mean["alpha_cell"]
        except AttributeError:
            self.plot.draw()
            return

        l = plot_density_hist(x, ax=self.plot.axes, color="C1")
        #stat_k = get_mode_stats(data.k_cell)
        self.plot.axes.set_xlim((1, 1))
        self.plot.axes.xaxis.set_ticks(np.arange(0, 1, 0.2))
        self.plot.axes.grid()
        self.plot.draw()


    def plot_kHist(self):
        self.plot.axes.clear()
        try:
            x = self.data_mean["k_cell"]
        except AttributeError:
            self.plot.draw()
            return

        l = plot_density_hist(np.log10(x), ax=self.plot.axes, color="C0")
        # stat_k = get_mode_stats(data.k_cell)
        self.plot.axes.set_xlim((1, 4))
        self.plot.axes.xaxis.set_ticks(np.arange(5))
        self.plot.axes.grid()
        self.plot.draw()

    def plot_k_alpha(self):
        self.plot_scatter("alpha_cell", "k_cell", funct2=np.log10)
        self.plot.axes.set_ylabel("log10 k")
        self.plot.axes.set_xlabel("alpha")
        self.plot.figure.tight_layout()
        self.plot.draw()


    def plot_irreg(self):
        # unfiltered plot of irregularity and solidity to easily identify errors
        # currently based on single cells

        self.plot_scatter("solidity", "irregularity")
        self.plot.axes.axvline(self.sol_threshold, ls="--")
        self.plot.axes.axhline(self.reg_threshold, ls="--")
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_stress_strain(self):
        self.plot_scatter("stress", "strain")
        self.plot.axes.set_xlim((-10, 400))
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_scatter(self, type1, type2, funct1=doNothing, funct2=doNothing):

        self.plot.axes.clear()
        try:
            x = funct1(self.data_mean[type1])
            y = funct2(self.data_mean[type2])
        except KeyError:
            self.plot.draw()
            return
        plotDensityScatter(x, y, cmap='viridis', alpha=1, skip=1, y_factor=1, s=5, levels=None, loglog=False,
                           ax=self.plot.axes)
        self.plot_data = np.array([x, y])
        self.plot.axes.set_xlabel(type1)
        self.plot.axes.set_ylabel(type2)


    def buttonPressedEvent(self):
        self.show()

    def initUnet(self):
        print("loading weight file: ", self.weight_selection.file)
        shape = self.cp.getImage().getShape()
        self.unet = UNet((shape[0], shape[1], 1), 1, d=8, weights=self.weight_selection.file)

    def start_threaded(self, run_function):
        self.stop=False
        self.thread.run_function = run_function
        self.thread.start()
    def quit_thread(self):
        self.stop=True
        self.thread.quit()


    def detect_all(self):
        print("cell detection frame %d to %d" % (self.cp.getFrameRange()[0], self.cp.getFrameRange()[1]))
        # TODO: put this in a qworker or something...
        self.data_all_new = pd.DataFrame()
        cells = [{}]
        self.db.deleteEllipses(type=self.marker_type_cell2)

        self.progressbar.setMinimum(self.cp.getFrameRange()[0])
        self.progressbar.setMaximum(self.cp.getFrameRange()[1])
        self.pbarLable.setText("cell detection frame %d to %d" % (self.cp.getFrameRange()[0], self.cp.getFrameRange()[1]))
        for frame in range(self.cp.getFrameRange()[0], self.cp.getFrameRange()[1]):
            if self.stop:
                break
            im = self.db.getImage(frame=frame)
            img = im.data
            cells, probability_map = self.detect(im, img, frame)

            for cell in cells:
                self.data_all_new = self.data_all_new.append(cell, ignore_index=True)
            self.progressbar.setValue(frame)
            if frame % 10 == 0:
                self.cp.reloadMask()
                self.cp.reloadMarker()

        self.cp.reloadMask()
        self.cp.reloadMarker()
        self.data_all_new["timestamp"] = self.data_all_new["timestamp"].astype(float)
        self.data_all_new["frames"] = self.data_all_new["frames"].astype(int)
        # save data
        save_cells_to_file(self.addon_result_file,  self.data_all_new.to_dict("records"))
        # getting the cell ids correctly
        print("tanktreading")
        self.tank_treading(self.data_all_new)
        print("evaluation")
        if self.addon_evaluated_file.exists():
            os.remove(self.addon_evaluated_file)
        self.data_all_new, self.data_mean_new = self.load_data(self.addon_result_file, self.sol_threshold, self.reg_threshold)
        self.progressbar.setValue(self.cp.getFrameRange()[1])
        self.pbarLable.setText("")

    def tank_treading(self, data):
        ## TODO implement tanktreading for non video database
        image_reader = CachedImageReader(str(self.filename))
        getVelocity(data, self.config)
        correctCenter(data, self.config)
        data = data[(data.solidity > self.sol_threshold) & (data.irregularity < self.reg_threshold)]
        ids = pd.unique(data["cell_id"])
        results = []
        for id in ids:
            d = data[data.cell_id == id]
            crops, shifts, valid = getCroppedImages(image_reader, d)
            if len(crops) <= 1:
                continue
            crops = crops[valid]
            time = (d.timestamp - d.iloc[0].timestamp) * 1e-3
            speed, r2 = doTracking(crops, data0=d, times=np.array(time), pixel_size=self.config["pixel_size"])
            results.append([id, speed, r2])
        data = pd.DataFrame(results, columns=["id", "tt", "tt_r2"])
        data.to_csv(self.filename[:-4] + "_addon_tt.csv")

    def detect_single(self):

        im = self.cp.getImage()
        img = self.cp.getImage().data
        frame = im.frame
        cells, probability_map = self.detect(im, img, frame)

        self.cp.reloadMask()
        self.cp.reloadMarker()

        # writing probability map as an addtional layer
        filename = os.path.join(self.prob_folder, "%dprob_map.tiff" % (frame))
        Image.fromarray((probability_map * 255).astype(np.uint8)).save(filename)
        try:
            self.db.setImage(filename=filename, sort_index=frame, layer=self.prob_layer, path=self.prob_path)
        except peewee.IntegrityError:
            pass



    def detect(self, im, img, frame):

        if self.unet is None:
            self.unet = UNet((img.shape[0], img.shape[1], 1), 1, d=8)
        img = (img - np.mean(img)) / np.std(img).astype(np.float32)
        timestamp = getTimestamp(self.vidcap, frame)

        probability_map = self.unet.predict(img[None, :, :, None])[0, :, :, 0]
        prediction_mask = probability_map > 0.5
        cells, prediction_mask = mask_to_cells_edge(prediction_mask, img, self.config, 0, {}, edge_dist=15, return_mask=True)
        [c.update({"frames": frame, "timestamp": timestamp}) for c in cells]  # maybe use map for this?

        self.db.setMask(image=im, data=prediction_mask.astype(np.uint8))
        self.db.deleteEllipses(type=self.marker_type_cell2, image=im) # delete everything in detect_all
        self.drawEllipse(pd.DataFrame(cells), self.marker_type_cell2)

        return cells, probability_map



    def keyPressEvent(self, event):

        # if event.key() == QtCore.Qt.Key_PageUp:
        #    self.detect_single()
        #    self.cp.window.layer_index = 2
       #     self.cp.jumpToFrame(self.cp.getCurrentFrame())

        if event.key() == QtCore.Qt.Key_G:
            print("detecting")
            self.detect_single()
            print("detecting finished")

  #  def LayerChangedEvent(self):
    #    pass
        #print("###")
       # if self.cp.getImage().layer.id == 0: # this needs to be base layer ...
        #    self.detect_single()

    def frameChangedEvent(self):
        pass
        # if self.segementation_tickbox.isChecked():
        #    self.detect_single() # ToDo: make this somehow optional
        # TODO Probably remove this completely??
        # if im is not None and self.data_all is not None and im.ellipses.count() == 0:
        #     for index, element in self.data_all[self.data_all.frames == im.frame].iterrows():
        #        self.set_ellipse(element, im)
        # TODO use cell type 1 again

    def drawEllipse(self, data_block, type):
        if len(data_block) == 0:
            return

        strains = (data_block["long_axis"] - data_block["short_axis"]) / np.sqrt(
            data_block["long_axis"] * data_block["short_axis"])
        text = []
        for s, sol, irr in zip(strains, data_block['solidity'], data_block['irregularity']):
            text.append(f"strain {s:.3f}\nsolidity {sol:.2f}\nirreg. {irr:.3f}")
        self.db.setEllipses(frame=list(data_block["frames"]), x=list(data_block["x"]),
                            y=list(data_block["y"]), width=list(data_block["long_axis"] / self.config["pixel_size"]),
                            height=list(data_block["short_axis"] / self.config["pixel_size"]),
                            angle=list(data_block["angle"]), type=type, text=text)


    def display_ellipses(self, type="cell", data=None):

        batch_size = 200
        data = data if not data is None else self.data_all_existing
        if len(data) == 0:
            return
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(len(data))
        self.pbarLable.setText("displaying ellipses")
        for block in range(0, len(data), batch_size):
            if self.stop:
                break
            if block + batch_size > len(data):
                data_block = data.iloc[block:]
            else:
                data_block = data.iloc[block:block + batch_size]

            self.drawEllipse(data_block, type)
            self.cp.reloadMarker()
            self.progressbar.setValue(block)
        self.progressbar.setValue(len(data))
        self.pbarLable.setText("")








# TODO: implement reg/sol thresholds
# TODO: custom network
# TODO multiple network prediction

"""
 def fill_with_mean(self, data, data_mean):
        # TODO discuss: what quantities should be averaged over --> nothing??
        # writing back to dataframe with all cells
       # mean_cols = ['long_axis', 'short_axis', 'angle',
        #             'irregularity', 'solidity', 'sharpness', 'velocity']
        mean_cols = ['velocity']
        # this is 1000 times faster
        data_cp = data_mean.set_index("cell_id")[mean_cols].to_dict("index")
        data_cp__ = data.to_dict("index") # TODO what about the filtered cells
        for index, cell_id in zip(data.index, data.cell_id):
            try:
                data_cp__[index].update(data_cp[cell_id])
            except KeyError:
                pass
        data_all = pd.DataFrame(data_cp__.values()) # note this is reiindexing

        return data_mean, data_all
"""









"""
    def plot_y_strain(self):
        y = self.data[:, 2]
        stress_values = stressfunc(self.data[:, 3] * 1e-6, self.config)
        strain_values = strain(self.data[:, 4], self.data[:, 5])

        self.plot.axes.clear()

        self.plot_data = np.array([y, strain_values])
        self.plot.axes.plot(y, strain_values, "o")
        self.plot.axes.set_xlabel("y")
        self.plot.axes.set_ylabel("strain")
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_y_angle(self):
        y = self.data[:, 2]
        angle = self.data[:, 6]

        self.plot.axes.clear()

        self.plot_data = np.array([y, angle])
        self.plot.axes.plot(y, angle, "o")
        self.plot.axes.set_xlabel("y")
        self.plot.axes.set_ylabel("angle")
        self.plot.figure.tight_layout()
        self.plot.draw()





from __future__ import division, print_function
import clickpoints
from clickpoints.includes.QtShortCuts import AddQComboBox, AddQSaveFileChoose, AddQSpinBox, AddQLineEdit
from qtpy import QtCore, QtGui, QtWidgets
import numpy as np
from clickpoints.includes.matplotlibwidget import MatplotlibWidget, NavigationToolbar
from matplotlib import pyplot as plt
import time
import configparser

import os
import sys
sys.path.insert(0, r"C:\Software\Deformation_Cytometer")
from helper_functions import getConfig


def getData(datafile):
    datafile = datafile.replace(".tif", "_result.txt")
    #%% import raw data
    data =np.genfromtxt(datafile,dtype=float,skip_header= 2)
    return data


def stressfunc(R,filename_config): # imputs (radial position and pressure)
    config = configparser.ConfigParser()
    config.read(filename_config) 

    pressure=float(config['SETUP']['pressure'].split()[0])*1000 #applied pressure (in Pa)
    channel_width=float(config['SETUP']['channel width'].split()[0])*1e-6 #in m
    #channel_width=196*1e-6 #in m
    channel_length=float(config['SETUP']['channel length'].split()[0])*1e-2 #in m
    framerate=float(config['CAMERA']['frame rate'].split()[0]) #in m
    
    magnification=float(config['MICROSCOPE']['objective'].split()[0])
    coupler=float(config['MICROSCOPE']['coupler'] .split()[0])
    camera_pixel_size=float(config['CAMERA']['camera pixel size'] .split()[0])
    
    pixel_size=camera_pixel_size/(magnification*coupler) # in micrometer
    
    #%% stress profile in channel
    L=channel_length #length of the microchannel in meter
    H= channel_width #height(and width) of the channel 
    
    P = -pressure

    G=P/L #  pressure gradient
    pre_factor=(4*(H**2)*G)/(np.pi)**3
    u_primy=np.zeros(len(R))  
    sumi=0
    for i in range(0,len(R)): 
        for n in range(1,100,2): # sigma only over odd numbers
            u_primey=pre_factor *  ((-1)**((n-1)/2))*(np.pi/((n**2)*H))\
            * (np.sinh((n*np.pi*R[i])/H)/np.cosh(n*np.pi/2))
            sumi=u_primey + sumi
        u_primy[i]=sumi
        sumi=0
    stress= np.sqrt((u_primy)**2)
    return stress #output atress profile


def strain(longaxis, shortaxis):
    D = np.sqrt(longaxis * shortaxis) #diameter of undeformed (circular) cell
    strain = (longaxis - shortaxis) / D
    return strain


class Addon(clickpoints.Addon):
    signal_update_plot = QtCore.Signal()
    signal_plot_finished = QtCore.Signal()
    image_plot = None
    last_update = 0
    updating = False
    exporting = False
    exporting_index = 0

    def __init__(self, *args, **kwargs):
        clickpoints.Addon.__init__(self, *args, **kwargs)
        # set the title and layout
        self.setWindowTitle("DeformationCytometer - ClickPoints")
        self.layout = QtWidgets.QVBoxLayout(self)

        # add export buttons
        layout = QtWidgets.QHBoxLayout()
        self.button_stressstrain = QtWidgets.QPushButton("stress-strain")
        self.button_stressstrain.clicked.connect(self.plot_stress_strain)
        layout.addWidget(self.button_stressstrain)

        self.button_stressy = QtWidgets.QPushButton("y-strain")
        self.button_stressy.clicked.connect(self.plot_y_strain)
        layout.addWidget(self.button_stressy)

        self.button_y_angle = QtWidgets.QPushButton("y-angle")
        self.button_y_angle.clicked.connect(self.plot_y_angle)
        layout.addWidget(self.button_y_angle)

        self.layout.addLayout(layout)

        # add a plot widget
        self.plot = MatplotlibWidget(self)
        self.layout.addWidget(self.plot)
        self.layout.addWidget(NavigationToolbar(self.plot, self))
        self.plot.figure.canvas.mpl_connect('button_press_event', self.button_press_callback)

        # add a progress bar
        self.progressbar = QtWidgets.QProgressBar()
        self.layout.addWidget(self.progressbar)

        # connect slots
        #self.signal_update_plot.connect(self.updatePlotImageEvent)
        #self.signal_plot_finished.connect(self.plotFinishedEvent)

        # initialize the table
        #self.updateTable()
        #self.selected = None

        filename = self.db.getImage(0).get_full_filename()
        print(filename.replace(".tif", "_config.txt"))
        self.config = getConfig(filename.replace(".tif", "_config.txt"))
        self.data = getData(filename)

    def button_press_callback(self, event):
        # only drag with left mouse button
        if event.button != 1:
            return
        # if the user doesn't have clicked on an axis do nothing
        if event.inaxes is None:
            return
        # get the pixel of the kymograph
        xy = np.array([event.xdata, event.ydata])
        scale = np.mean(self.plot_data, axis=1)
        distance = np.linalg.norm(self.plot_data/scale[:, None] - xy[:, None]/scale[:, None], axis=0)
        print(self.plot_data.shape, xy[:, None].shape, distance.shape)
        nearest_dist = np.min(distance)
        print("distance ", nearest_dist)
        nearest_point = np.argmin(distance)

        filename = self.db.getImage(0).get_full_filename()
        stress_values = stressfunc(self.data[:, 3] * 1e-6, filename.replace(".tif", "_config.txt"))
        strain_values = strain(self.data[:, 4], self.data[:, 5])

        print(np.linalg.norm(np.array([stress_values[nearest_point], strain_values[nearest_point]]) - xy))

        print("clicked", xy, stress_values[nearest_point], " ", strain_values[nearest_point], self.data[nearest_point])

        #x, y = event.xdata/self.input_scale1.value(), event.ydata/self.h/self.input_scale2.value()
        # jump to the frame in time
        self.cp.jumpToFrame(self.data[self.index][nearest_point, 0])
        # and to the xy position
        self.cp.centerOn(self.data[self.index][nearest_point, 1], self.data[self.index][nearest_point, 2])

    def plot_stress_strain(self):
        filename = self.db.getImage(0).get_full_filename()
        stress_values = stressfunc(self.data[:, 3]*1e-6, filename.replace(".tif", "_config.txt"))
        strain_values = strain(self.data[:,4], self.data[:,5])

        Irregularity = self.data[:, 7]  # ratio of circumference of the binarized image to the circumference of the ellipse
        Solidity = self.data[:, 8]  # percentage of binary pixels within convex hull polygon

        self.index = Solidity > 0#(Solidity > 0.96) & (Irregularity < 1.05)

        self.plot.axes.clear()

        self.plot_data = np.array([stress_values[self.index], strain_values[self.index]])
        self.plot.axes.plot(stress_values, strain_values, "o")
        self.plot.axes.set_xlabel("stress")
        self.plot.axes.set_ylabel("strain")
        self.plot.axes.set_xlim(-10, 400)
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_y_strain(self):
        y = self.data[:, 2]
        stress_values = stressfunc(self.data[:, 3]*1e-6, self.config)
        strain_values = strain(self.data[:,4], self.data[:,5])

        self.plot.axes.clear()

        self.plot_data = np.array([y, strain_values])
        self.plot.axes.plot(y, strain_values, "o")
        self.plot.axes.set_xlabel("y")
        self.plot.axes.set_ylabel("strain")
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_y_angle(self):
        y = self.data[:, 2]
        angle = self.data[:, 6]

        self.plot.axes.clear()

        self.plot_data = np.array([y, angle])
        self.plot.axes.plot(y, angle, "o")
        self.plot.axes.set_xlabel("y")
        self.plot.axes.set_ylabel("angle")
        self.plot.figure.tight_layout()
        self.plot.draw()


    def export(self):
        pass

    def buttonPressedEvent(self):
        self.show()
"""

