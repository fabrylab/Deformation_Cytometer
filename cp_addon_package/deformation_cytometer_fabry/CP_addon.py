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
# but WITHOUT ANY WARRANTY; without even the implied warranty of
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
from deformation_cytometer_fabry.UNETmodel import UNet
from pathlib import Path



class Addon(clickpoints.Addon):
    data = None
    unet = None

    def __init__(self, *args, **kwargs):
        clickpoints.Addon.__init__(self, *args, **kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)

        # Check if the marker type is present
        self.marker_type_cell = self.db.setMarkerType("cell", "#0a2eff", self.db.TYPE_Ellipse)
        self.marker_type_cell2 = self.db.setMarkerType("cell2", "#Fa2eff", self.db.TYPE_Ellipse)
        self.cp.reloadTypes()

        self.loadData()

    def detect(self):
        if self.unet is None:
            self.unet = UNet().create_model((720, 540, 1), 1, d=8)

            # change path for weights
            self.unet.load_weights(str(Path(__file__).parent / "Unet_0-0-5_fl_RAdam_20200610-141144.h5"))
        im = self.cp.getImage()
        img = self.cp.getImage().data
        img = (img - np.mean(img)) / np.std(img).astype(np.float32)
        prediction_mask = self.unet.predict(img[None, :, :, None])[0, :, :, 0] > 0.5
        self.db.setMask(image=self.cp.getImage(), data=prediction_mask.astype(np.uint8))
        print(prediction_mask.shape)
        self.cp.reloadMask()
        print(prediction_mask)

        labeled = label(prediction_mask)

        # iterate over all detected regions
        for region in regionprops(labeled, img):
            y, x = region.centroid
            if region.orientation > 0:
                ellipse_angle = np.pi / 2 - region.orientation
            else:
                ellipse_angle = -np.pi / 2 - region.orientation
            self.db.setEllipse(image=im, x=x, y=y, width=region.major_axis_length, height=region.minor_axis_length,
                               angle=ellipse_angle, type=self.marker_type_cell2)

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

            self.data = np.genfromtxt(im.filename.replace(".tif", "_result.txt"), dtype=float, skip_header=2)
            self.frames = self.data[:, 0].astype("int")

    def frameChangedEvent(self):
        self.loadData()
        im = self.cp.getImage()
        if im is not None and self.data is not None and im.ellipses.count() == 0:
            for element in self.data[self.frames == im.frame]:
                x_pos = element[1]
                y_pos = element[2]
                long = element[4]
                short = element[5]
                angle = element[6]

                Irregularity = element[7]  # ratio of circumference of the binarized image to the circumference of the ellipse
                Solidity = element[8]  # percentage of binary pixels within convex hull polygon

                D = np.sqrt(long * short)  # diameter of undeformed (circular) cell
                strain = (long - short) / D

                self.db.setEllipse(image=im, x=x_pos, y=y_pos, width=long/self.pixel_size, height=short/self.pixel_size, angle=angle, type=self.marker_type_cell,
                                   text="strain %.3f\nsolidity %.2f\nirreg. %.3f"%(strain, Solidity, Irregularity))


    def buttonPressedEvent(self):
        self.show()


"""

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