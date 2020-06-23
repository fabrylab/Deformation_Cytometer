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

                self.db.setEllipse(image=im, x=x_pos, y=y_pos, width=long/self.pixel_size, height=short/self.pixel_size, angle=angle, type=self.marker_type_cell)


    def buttonPressedEvent(self):
        self.show()

