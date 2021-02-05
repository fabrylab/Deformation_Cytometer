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

import numpy as np
import qtawesome as qta
from qtpy import QtCore, QtWidgets

import clickpoints
from clickpoints.includes.QtShortCuts import AddQSpinBox, AddQOpenFileChoose
from clickpoints.includes import QtShortCuts

from inspect import getdoc
import traceback
import os
import sys
from importlib import import_module, reload
import configparser
from skimage.measure import label, regionprops
from deformationcytometer.detection.includes.UNETmodel import UNet
from deformationcytometer.detection.includes.regionprops import mask_to_cells_edge
from pathlib import Path

import numpy as np
from clickpoints.includes.matplotlibwidget import MatplotlibWidget, NavigationToolbar
from matplotlib import pyplot as plt
import time
from pathlib import Path
print(Path(__file__).parent.parent.parent)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deformationcytometer.includes.includes import getConfig
from deformationcytometer.includes.includes import getInputFile, getConfig, getData
from deformationcytometer.evaluation.helper_functions import getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from deformationcytometer.evaluation.helper_functions import plotDensityScatter
from deformationcytometer.detection.includes.UNETmodel import store_path
import deformationcytometer.detection.includes.UNETmodel

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
from scipy.stats import gaussian_kde
def plotDensityScatter(x, y, ax):
    x = x.to_numpy()
    y = y.to_numpy()
    xy = np.vstack([x, y])
    kd = gaussian_kde(xy)(xy)
    idx = kd.argsort()
    x, y, z = x[idx], y[idx], kd[idx]
    ax.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap='viridis')  # plot in kernel density colors e.g. viridis





from PyQt5.QtCore import pyqtSignal
from functools import partial
from deformationcytometer.includes.includes import Dialog
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


        self.unet = None
        self.layout = QtWidgets.QVBoxLayout(self)


        # Check if the marker type is present
        self.marker_type_cell = self.db.setMarkerType("cell", "#0a2eff", self.db.TYPE_Ellipse)
        self.marker_type_cell2 = self.db.setMarkerType("cell2", "#Fa2eff", self.db.TYPE_Ellipse)
        self.cp.reloadTypes()


        clickpoints.Addon.__init__(self, *args, **kwargs)

        # set the title and layout
        self.setWindowTitle("DeformationCytometer - ClickPoints")
        self.layout = QtWidgets.QVBoxLayout(self)
        # weight file selection
        self.weight_selection = SetFile(store_path, filetype="weight file (*.h5)")
        self.weight_selection.fileSeleted.connect(self.initUnet)
        self.layout.addLayout(self.weight_selection)
        # update segmentation
        self.update_detection_button = QtWidgets.QPushButton("update_detection")
        self.update_detection_button.clicked.connect(self.detect_all)
        self.layout.addWidget(self.update_detection_button)


        # add export buttons
        layout = QtWidgets.QHBoxLayout()
        self.button_stressstrain = QtWidgets.QPushButton("stress-strain")
        self.button_stressstrain.clicked.connect(self.plot_stress_strain)
        layout.addWidget(self.button_stressstrain)

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
        # TODO use this progress bar
        self.progressbar = QtWidgets.QProgressBar()
        self.layout.addWidget(self.progressbar)


        # connect slots
        # self.signal_update_plot.connect(self.updatePlotImageEvent)
        # self.signal_plot_finished.connect(self.plotFinishedEvent)

        # initialize the table
        # self.updateTable()
        # self.selected = None


        filename = self.db.getImage(0).get_full_filename()
        config_file = Path(filename.replace(".tif", "_config.txt"))
        result_file = Path(filename.replace(".tif", "_result.txt"))


        print(filename.replace(".tif", "_config.txt"))
        if Path(config_file).exists() and result_file.exist():
            self.loadData()
            self.config = getConfig(config_file)
            self.data = getData(result_file)
           # evaluation...
            #getVelocity(self.data, self.config) TODO reimplement that in a better way

            try:
                correctCenter(self.data, self.config)
            except ValueError:
                pass

           # self.data = self.data.groupby(['cell_id']).mean()
            self.sol_threshold = 0.7
            self.reg_threshold = 1.3
            self.data = filterCells(self.data, self.config, solidity_threshold=self.sol_threshold, irregularity_threshold=self.reg_threshold)
            self.data.reset_index(drop=True, inplace=True)

            getStressStrain(self.data, self.config)
        else:
            self.config = getConfig(default_config_path) # TODO fix this// save a default config somewhere

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

        filename = self.db.getImage(0).get_full_filename()
        stress_values = stressfunc(self.data.iloc[:, 3] * 1e-6, filename.replace(".tif", "_config.txt"))
        strain_values = strain(self.data.iloc[:, 4], self.data.iloc[:, 5])

        print(np.linalg.norm(np.array([stress_values[nearest_point], strain_values[nearest_point]]) - xy))

        print("clicked", xy, stress_values[nearest_point], " ", strain_values[nearest_point], self.data.iloc[nearest_point])

        # x, y = event.xdata/self.input_scale1.value(), event.ydata/self.h/self.input_scale2.value()
        # jump to the frame in time
        self.cp.jumpToFrame(self.data.frames[nearest_point])
        # and to the xy position
        self.cp.centerOn(self.data.x[nearest_point], self.data.y[nearest_point])

    def plot_stress_strain(self):
        filename = self.db.getImage(0).get_full_filename()

        self.plot.axes.clear()

        #plt.sca(self.plot.axes)
        x = self.data.stress
        y = self.data.strain
        plotDensityScatter(x, y, ax=self.plot.axes)

        self.plot_data = np.array([x, y])
        #self.plot.axes.plot(stress_values, strain_values, "o")
        self.plot.axes.set_xlabel("stress")
        self.plot.axes.set_ylabel("strain")
        self.plot.axes.set_xlim(-10, 400)
        self.plot.figure.tight_layout()
        self.plot.draw()

    def export(self):
        pass

    def buttonPressedEvent(self):
        self.show()

    def initUnet(self):
        print("loading weight file: ", self.weight_selection.file)
        shape = self.cp.getImage().getShape()
        self.unet = UNet((shape[0], shape[1], 1), 1, d=8, weights=self.weight_selection.file)

    def detect_all(self):
        # TODO: put this in a qworker or something...
        frames = self.db.getImageCount()
        self.progressbar.setMaximum(frames)
        for frame in range(frames):
            im = self.db.getImage(frame=frame)
            img = im.data
            self.detect(im, img)
            self.progressbar.setValue(frame)
        self.cp.reloadMask() # TODO move into loop?
        self.cp.reloadMarker()

    def detect_single(self):
        im = self.cp.getImage()
        img = self.cp.getImage().data
        self.detect(im, img)

        self.cp.reloadMask()
        self.cp.reloadMarker()

    def detect(self, im, img):

        if self.unet is None:
            self.unet = UNet((img.shape[0], img.shape[1], 1), 1, d=8)
        img = (img - np.mean(img)) / np.std(img).astype(np.float32)
        prediction_mask = self.unet.predict(img[None, :, :, None])[0, :, :, 0] > 0.5
        cells, prediction_mask = mask_to_cells_edge(prediction_mask, img, self.config, 0, {}, edge_dist=15, return_mask=True)
        self.db.setMask(image=self.cp.getImage(), data=prediction_mask.astype(np.uint8))
        self.db.deleteEllipses(type=self.marker_type_cell2, frame=self.cp.getCurrentFrame())
        for cell in cells:
            strain = (cell["long_axis"] - cell["short_axis"]) / np.sqrt(cell["long_axis"] * cell["short_axis"])
            self.db.setEllipse(image=im, x=cell["x_pos"], y=cell["y_pos"], width=cell["long_axis"]/self.config["pixel_size"], height=cell["short_axis"]/self.config["pixel_size"],
                               angle=cell["angle"], type=self.marker_type_cell2, text=
                               f"strain {strain:.3f}\nsolidity {cell['solidity']:.2f}\nirreg. {cell['irregularity']:.3f}")




    def keyPressEvent(self, event):
        print(event.key(), QtCore.Qt.Key_G)
        if event.key() == QtCore.Qt.Key_G:
            print("detect")
            self.detect()

    def loadData(self):
        if self.data is not None:
            return
        im = self.cp.getImage()
        if im is not None:
            config = configparser.ConfigParser()
            config.read(im.filename.replace(".tif", "_config.txt"))

            magnification = float(config['MICROSCOPE']['objective'].split()[0])
            coupler = float(config['MICROSCOPE']['coupler'].split()[0])
            camera_pixel_size = float(config['CAMERA']['camera pixel size'].split()[0])

            self.pixel_size = camera_pixel_size / (magnification * coupler)  # in micrometer

            self.data2 = np.genfromtxt(im.filename.replace(".tif", "_result.txt"), dtype=float, skip_header=2)
            self.frames = self.data2[:, 0].astype("int")

    def frameChangedEvent(self):
        self.loadData()
        im = self.cp.getImage()
        if im is not None and self.data is not None and im.ellipses.count() == 0:
            for index, element in self.data[self.data.frames == im.frame].iterrows():
                print("element")
                x_pos = element.x
                y_pos = element.y
                long = element.long_axis
                short = element.short_axis
                angle = element.angle

                Irregularity = element.irregularity  # ratio of circumference of the binarized image to the circumference of the ellipse
                Solidity = element.solidity  # percentage of binary pixels within convex hull polygon

                D = np.sqrt(long * short)  # diameter of undeformed (circular) cell
                strain = (long - short) / D

                #print("element.velocity_partner", element.velocity_partner)

                self.db.setEllipse(image=im, x=x_pos, y=y_pos, width=long/self.pixel_size, height=short/self.pixel_size, angle=angle, type=self.marker_type_cell,
                                   text=f"timestamp {element.timestamp}\nstrain {strain:.3f}\nsolidity {Solidity:.2f}\nirreg. {Irregularity:.3f}",#\nvelocity {element.velocity:.3f}\n {element.velocity_partner}"
                                   )






# TODO: implement reg/sol thresholds
# TODO: custom network
# TODO multiple network prediktion



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

