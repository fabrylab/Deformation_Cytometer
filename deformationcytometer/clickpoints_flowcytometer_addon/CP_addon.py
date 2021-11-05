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
from qtpy import QtCore, QtWidgets, QtGui
import imageio
import shutil
import clickpoints
import peewee
import os
import pandas as pd
import numpy as np
from clickpoints.includes.matplotlibwidget import MatplotlibWidget, NavigationToolbar
from PyQt5.QtCore import pyqtSlot
from functools import partial
from PIL import Image
from deformationcytometer.tanktreading.helpers import getCroppedImages, doTracking, CachedImageReader
from deformationcytometer.detection.includes.UNETmodel import UNet, store_path
from deformationcytometer.detection.includes.regionprops import mask_to_cells_edge, getTimestamp, save_cells_to_file
from deformationcytometer.includes.includes import getConfig, getData
from deformationcytometer.evaluation.helper_functions import getVelocity, correctCenter, plotDensityScatter, \
    load_all_data, plot_density_hist, load_all_data_new
from deformationcytometer.clickpoints_flowcytometer_addon.includes import *


class Addon(clickpoints.Addon):
    signal_update_plot = QtCore.Signal()
    signal_plot_finished = QtCore.Signal()
    disp_text_existing = "displaying existing data"
    disp_text_new = "displaying new data"

    def __init__(self, *args, **kwargs):
        clickpoints.Addon.__init__(self, *args, **kwargs)

        # qthread and signals for update cell detection and loading ellipse at add on launch
        self.thread = Worker(run_function=None)
        self.thread.thread_started.connect(self.start_pbar)
        self.thread.thread_finished.connect(self.finish_pbar)
        self.thread.thread_progress.connect(self.update_pbar)

        self.stop = False
        self.plot_data = np.array([[], []])
        self.unet = None
        self.layout = QtWidgets.QVBoxLayout(self)

        # Setting up marker Types
        self.marker_type_cell1 = self.db.setMarkerType("cell", "#0a2eff", self.db.TYPE_Ellipse)
        self.marker_type_cell2 = self.db.setMarkerType("cell new", "#Fa2eff", self.db.TYPE_Ellipse)
        self.cp.reloadTypes()

        # finding and setting path to store network probability map
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
        # in range of frames
        seg_layout = QtWidgets.QHBoxLayout()
        self.update_detection_button = QtWidgets.QPushButton("update cell detection")
        self.update_detection_button.setToolTip(tooltip_strings["update cell detection"])
        self.update_detection_button.clicked.connect(partial(self.start_threaded, self.detect_all))
        seg_layout.addWidget(self.update_detection_button, stretch=5)
        # on single frame
        self.update_single_detection_button = QtWidgets.QPushButton("single detection")
        self.update_single_detection_button.setToolTip(tooltip_strings["single detection"])
        self.update_single_detection_button.clicked.connect(self.detect_single)
        seg_layout.addWidget(self.update_single_detection_button, stretch=1)
        self.layout.addLayout(seg_layout)

        # regularity and solidity thresholds
        validator = QtGui.QDoubleValidator(0, 100, 3)
        filter_layout = QtWidgets.QHBoxLayout()
        reg_label = QtWidgets.QLabel("irregularity")
        filter_layout.addWidget(reg_label)
        self.reg_box = QtWidgets.QLineEdit("1.06")
        self.reg_box.setToolTip(tooltip_strings["irregularity"])
        self.reg_box.setValidator(validator)
        filter_layout.addWidget(self.reg_box, stretch=1)  # TODO implement text edited method
        sol_label = QtWidgets.QLabel("solidity")
        filter_layout.addWidget(sol_label)
        self.sol_box = QtWidgets.QLineEdit("0.96")
        self.sol_box.setToolTip(tooltip_strings["solidity"])
        self.sol_box.setValidator(validator)
        filter_layout.addWidget(self.sol_box, stretch=1)
        rmin_label = QtWidgets.QLabel("min radius [µm]")
        filter_layout.addWidget(rmin_label)
        self.rmin_box = QtWidgets.QLineEdit("6")
        self.rmin_box.setToolTip(tooltip_strings["min radius"])
        self.rmin_box.setValidator(validator)
        filter_layout.addWidget(self.rmin_box, stretch=1)
        filter_layout.addStretch(stretch=4)
        self.layout.addLayout(filter_layout)

        # plotting buttons
        layout = QtWidgets.QHBoxLayout()
        self.button_stressstrain = QtWidgets.QPushButton("stress-strain")
        self.button_stressstrain.clicked.connect(self.plot_stress_strain)
        self.button_stressstrain.setToolTip(tooltip_strings["stress-strain"])
        layout.addWidget(self.button_stressstrain)
        self.button_kpos = QtWidgets.QPushButton("k-pos")
        self.button_kpos.clicked.connect(self.plot_k_pos)
        self.button_kpos.setToolTip(tooltip_strings["k-pos"])
        layout.addWidget(self.button_kpos)
        self.button_reg_sol = QtWidgets.QPushButton("regularity-solidity")
        self.button_reg_sol.clicked.connect(self.plot_irreg)
        self.button_reg_sol.setToolTip(tooltip_strings["regularity-solidity"])
        layout.addWidget(self.button_reg_sol)
        self.button_kHist = QtWidgets.QPushButton("k histogram")
        self.button_kHist.clicked.connect(self.plot_kHist)
        self.button_kHist.setToolTip(tooltip_strings["k histogram"])
        layout.addWidget(self.button_kHist)
        self.button_alphaHist = QtWidgets.QPushButton("alpha histogram")
        self.button_alphaHist.clicked.connect(self.plot_alphaHist)
        self.button_alphaHist.setToolTip(tooltip_strings["alpha histogram"])
        layout.addWidget(self.button_alphaHist)
        self.button_kalpha = QtWidgets.QPushButton("k-alpha")
        self.button_kalpha.clicked.connect(self.plot_k_alpha)
        self.button_kalpha.setToolTip(tooltip_strings["k-alpha"])
        layout.addWidget(self.button_kalpha)
        # button to switch between display of loaded and newly generated data
        frame = QtWidgets.QFrame()  # horizontal separating line
        frame.setFrameShape(QtWidgets.QFrame.VLine)
        frame.setLineWidth(3)
        layout.addWidget(frame)
        self.switch_data_button = QtWidgets.QPushButton(self.disp_text_existing)
        self.switch_data_button.clicked.connect(self.switch_display_data)
        self.switch_data_button.setToolTip(tooltip_strings[self.disp_text_existing])
        layout.addWidget(self.switch_data_button)
        self.layout.addLayout(layout)

        # matplotlib widgets to draw plots
        self.plot = MatplotlibWidget(self)
        self.plot_data = np.array([[], []])
        self.layout.addWidget(self.plot)
        self.layout.addWidget(NavigationToolbar(self.plot, self))
        self.plot.figure.canvas.mpl_connect('button_press_event', self.button_press_callback)

        # progress bar
        self.progressbar = QtWidgets.QProgressBar()
        self.layout.addWidget(self.progressbar)
        # progressbar lable
        pbar_info_layout = QtWidgets.QHBoxLayout()
        self.pbarLable = QtWidgets.QLabel("")
        pbar_info_layout.addWidget(self.pbarLable, stretch=1)
        pbar_info_layout.addStretch(stretch=2)
        # button to stop thread execution
        self.stop_button = QtWidgets.QPushButton("stop")
        self.stop_button.clicked.connect(self.quit_thread)
        self.stop_button.setToolTip(tooltip_strings["stop"])
        pbar_info_layout.addWidget(self.stop_button, stretch=1)
        self.layout.addLayout(pbar_info_layout)

        # setting paths for data, config and image
        # identifying the full path to the video. If an existing ClickPoints database is opened, the path if
        # is likely relative to the database location.
        self.filename = self.db.getImage(0).get_full_filename()
        if not os.path.isabs(self.filename):
            self.filename = str(
                Path(self.db._database_filename).parent.joinpath(Path(self.filename)))

        self.config_file = self.constructFileNames("_config.txt")
        self.result_file = self.constructFileNames("_evaluated_new.csv")
        #self.addon_result_file = self.constructFileNames("_addon_result.txt")
        self.addon_evaluated_file = self.constructFileNames("_addon_evaluated.csv")
        self.addon_config_file = self.constructFileNames("_addon_config.txt")
        self.vidcap = imageio.get_reader(self.filename)

        # reading in config an data
        self.data_all_existing = pd.DataFrame()
        self.data_all_existing = pd.DataFrame()
        self.data_all_new = pd.DataFrame()
        self.data_all_new = pd.DataFrame()

        if self.config_file.exists() and self.result_file.exists():
            self.config = getConfig(self.config_file)
            # ToDo: replace with a flag// also maybe some sort of "reculation" feature
            # Trying to get regularity and solidity from the config
            if "irregularity" in self.config.keys() and "solidity" in self.config.keys():
                solidity_threshold = self.config["solidity"]
                irregularity_threshold = self.config["irregularity"]
            else:
                solidity_threshold = self.sol_threshold
                irregularity_threshold = self.reg_threshold
            # reading data from evaluated.csv
            self.data_all_existing = self.load_data(self.result_file, solidity_threshold, irregularity_threshold)
        else:  # get a default config if no config is found
            self.config = getConfig(default_config_path)

        ## loading data from previous addon action
        if self.addon_evaluated_file.exists():
            self.data_all_new = self.load_data(self.addon_evaluated_file, self.sol_threshold, self.reg_threshold)
            self.start_threaded(partial(self.display_ellipses, type=self.marker_type_cell2, data=self.data_all_new))
        # create an addon config file
        # presence of this file allows easy implementation of the load_data and tank threading pipelines when
        # calculating new data
        if not self.addon_config_file.exists():
            shutil.copy(self.config_file, self.addon_config_file)

        self.plot_data_frame = self.data_all
        # initialize plot
        self.plot_stress_strain()

        # Displaying the loaded cells. This is in separate thread as it takes up to 20 seconds.
        self.db.deleteEllipses(type=self.marker_type_cell1)
        self.db.deleteEllipses(type=self.marker_type_cell2)
        self.start_threaded(partial(self.display_ellipses, type=self.marker_type_cell1, data=self.data_all_existing))

        print("loading finished")

    def constructFileNames(self, replace):
        if self.filename.endswith(".tif"):
            return Path(self.filename.replace(".tif", replace))
        if self.filename.endswith(".cdb"):
            return Path(self.filename.replace(".cdb", replace))

    # slots to update the progress bar from another thread (update cell detection and display_ellipse)
    @pyqtSlot(tuple, str)  # the decorator is not really necessary
    def start_pbar(self, prange, text):
        self.progressbar.setMinimum(prange[0])
        self.progressbar.setMaximum(prange[1])
        self.pbarLable.setText(text)

    @pyqtSlot(int)
    def update_pbar(self, value):
        self.progressbar.setValue(value)

    @pyqtSlot(int)
    def finish_pbar(self, value):
        self.progressbar.setValue(value)
        self.pbarLable.setText("finished")

    # Dynamic switch between existing and new data
    def switch_display_data(self):

        if self.switch_data_button.text() == self.disp_text_existing:
            text = self.disp_text_new
        else:
            text = self.disp_text_existing
        self.switch_data_button.setText(text)
        # updating the plot
        self.plot_type()

    @property
    def data_all(self):
        if self.switch_data_button.text() == self.disp_text_existing:
            return self.data_all_existing
        if self.switch_data_button.text() == self.disp_text_new:
            return self.data_all_new

    @property
    def data_mean(self):
        if self.switch_data_button.text() == self.disp_text_existing:
            return self.data_all_existing
        if self.switch_data_button.text() == self.disp_text_new:
            return self.data_all_new

    # solidity and regularity and rmin properties
    @property
    def sol_threshold(self):
        return float(self.sol_box.text())

    @property
    def reg_threshold(self):
        return float(self.reg_box.text())

    @property
    def rmin(self):
        return float(self.rmin_box.text())

    # handling thread entrance and exit
    def start_threaded(self, run_function):
        self.stop = False  # self.stop property is used to by the thread function to exit loops
        self.thread.run_function = run_function
        self.thread.start()

    def quit_thread(self):
        self.stop = True
        self.thread.quit()

    def load_data(self, file, solidity_threshold, irregularity_threshold):

        #data_all = getData(file)
        #if not "area" in data_all.keys():
        #    data_all["area"] = data_all["long_axis"] * data_all["short_axis"] * np.pi/4

        #if len(data_all) == 0:
        #    print("no data loaded from file '%s'" % file)
        #    return pd.DataFrame(), pd.DataFrame()
        # use a "read sol from config flag here
        print("load_all_data_new", self.db.getImage(0).get_full_filename().replace(".tif", "_evaluated_new.csv"))
        data_mean, config_eval = load_all_data_new(self.db.getImage(0).get_full_filename().replace(".tif", "_evaluated_new.csv"), do_group=False, do_excude=False)
        return data_mean

    # plotting functions
    # wrapper for all scatter plots; handles empty and data log10 transform
    def plot_scatter(self, data, type1, type2, funct1=doNothing, funct2=doNothing):
        self.init_newPlot()
        try:
            x = funct1(data[type1])
            y = funct2(data[type2])
        except KeyError:
            self.plot.draw()
            return
        if (np.all(np.isnan(x))) or (np.all(np.isnan(x))):
            return
        try:
            plotDensityScatter(x, y, cmap='viridis', alpha=1, skip=1, y_factor=1, s=5, levels=None, loglog=False,
                               ax=self.plot.axes)
            self.plot_data = np.array([x, y])
            self.plot_data_frame = data
            self.plot.axes.set_xlabel(type1)
            self.plot.axes.set_ylabel(type2)
        except (ValueError, np.LinAlgError):
            print("kernel density estimation failed? not enough cells found?")
            return

    # clearing axis and plot.data
    def init_newPlot(self):
        self.plot_data = np.array([[], []])
        self.plot.axes.clear()
        self.plot.draw()

    def plot_alphaHist(self):
        self.plot_type = self.plot_alphaHist
        self.init_newPlot()
        try:
            x = self.data_mean["alpha_cell"]
        except KeyError:
            return
        if not np.any(~np.isnan(x)):
            return
        l = plot_density_hist(x, ax=self.plot.axes, color="C1")
        # stat_k = get_mode_stats(data.k_cell)
        self.plot.axes.set_xlim((1, 1))
        self.plot.axes.xaxis.set_ticks(np.arange(0, 1, 0.2))
        self.plot.axes.grid()
        self.plot.draw()

    def plot_kHist(self):
        self.plot_type =  self.plot_kHist
        self.init_newPlot()
        try:
            x = np.array(self.data_mean["k_cell"])
        except KeyError:
            return
        if not np.any(~np.isnan(x)):
            return
        l = plot_density_hist(np.log10(x), ax=self.plot.axes, color="C0")
        self.plot.axes.set_xlim((1, 4))
        self.plot.axes.xaxis.set_ticks(np.arange(5))
        self.plot.axes.grid()
        self.plot.draw()

    def plot_k_alpha(self):
        self.plot_type = self.plot_k_alpha
        self.plot_scatter(self.data_mean, "alpha_cell", "k_cell", funct2=np.log10)
        self.plot.axes.set_ylabel("log10 k")
        self.plot.axes.set_xlabel("alpha")
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_k_size(self):
        self.plot_type = self.plot_k_size
        self.plot_scatter(self.data_mean, "area", "w_k_cell") # use self.data_all for unfiltered data
        self.plot.axes.set_ylabel("w_k")
        self.plot.axes.set_xlabel("area")
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_k_pos(self):
        self.plot_type = self.plot_k_pos
        self.plot_scatter(self.data_mean, "rp", "w_k_cell") # use self.data_all for unfiltered data
        self.plot.axes.set_ylabel("w_k")
        self.plot.axes.set_xlabel("radiale position")
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_irreg(self):
        self.plot_type = self.plot_irreg
        # unfiltered plot of irregularity and solidity to easily identify errors
        # currently based on single cells
        self.plot_scatter(self.data_all, "solidity", "irregularity", funct1=doNothing, funct2=doNothing)
        self.plot.axes.axvline(self.sol_threshold, ls="--")
        self.plot.axes.axhline(self.reg_threshold, ls="--")
        self.plot.figure.tight_layout()
        self.plot.draw()

    def plot_stress_strain(self):
        self.plot_type = self.plot_stress_strain
        self.plot_scatter(self.data_mean, "stress", "strain")
        self.plot.axes.set_xlim((-10, 400))
        self.plot.figure.tight_layout()
        self.plot.draw()

    # Jump to cell in ClickPoints window when clicking near a data point in the scatter plot
    def button_press_callback(self, event):
        # only drag with left mouse button, do nothing if plot is empty or clicked outside of axis
        if event.button != 1 or event.inaxes is None or self.plot_data.size == 0:
            return
        xy = np.array([event.xdata, event.ydata])
        scale = np.nanmean(self.plot_data, axis=1)
        distance = np.linalg.norm(self.plot_data / scale[:, None] - xy[:, None] / scale[:, None], axis=0)
        nearest_point = np.nanargmin(distance)
        print("clicked", xy)
        self.cp.jumpToFrame(int(self.plot_data_frame.frames[nearest_point]))
        self.cp.centerOn(self.plot_data_frame.x[nearest_point], self.plot_data_frame.y[nearest_point])

    # not sure what this is for ^^
    def buttonPressedEvent(self):
        self.show()

    ## cell detection
    def initUnet(self):
        print("loading weight file: ", self.weight_selection.file)
        shape = self.cp.getImage().getShape()
        self.unet = UNet((shape[0], shape[1], 1), 1, d=8, weights=self.weight_selection.file)

    # cell detection and evaluation on multiple frames
    def detect_all(self):
        info = "cell detection frame %d to %d" % (self.cp.getFrameRange()[0], self.cp.getFrameRange()[1])
        print(info)

        self.data_all_new = pd.DataFrame()
        self.data_all_new = pd.DataFrame()
        self.db.deleteEllipses(type=self.marker_type_cell2)
        self.thread.thread_started.emit(tuple(self.cp.getFrameRange()[:2]), info)
        for frame in range(self.cp.getFrameRange()[0], self.cp.getFrameRange()[1]):
            if self.stop:  # stop signal from "stop" button
                break
            im = self.db.getImage(frame=frame)
            img = im.data
            cells, probability_map = self.detect(im, img, frame)
            for cell in cells:
                self.data_all_new = self.data_all_new.append(cell, ignore_index=True)
            self.thread.thread_progress.emit(frame)
            # reloading the mask and ellipse display in ClickPoints// may not be necessary to do it in batches
            if frame % 10 == 0:
                self.cp.reloadMask()
                self.cp.reloadMarker()

        self.cp.reloadMask()
        self.cp.reloadMarker()
        self.data_all_new["timestamp"] = self.data_all_new["timestamp"].astype(float)
        self.data_all_new["frames"] = self.data_all_new["frames"].astype(int)
        # save data to addon_result.txt file
        self.data_all_new.to_csv(self.addon_evaluated_file, index=False)
        #save_cells_to_file(self.addon_result_file, self.data_all_new.to_dict("records"))
        # tank threading
        print("tank threading")
        # catching error if no velocities could be identified (e.g. when only few cells are identified)
        try:
            self.tank_treading(self.data_all_new)
            # further evaluation
            print("evaluation")
            #if self.addon_evaluated_file.exists():
            #    os.remove(self.addon_evaluated_file)
            self.data_all_new = self.load_data(self.addon_evaluated_file, self.sol_threshold, self.reg_threshold)
        except ValueError as e:
            print(e)
        self.thread.thread_finished.emit(self.cp.getFrameRange()[1])
        print("finished")

    # tank threading: saves results to an "_addon_tt.csv" file
    def tank_treading(self, data):
        # TODO implement tank threading for non video database
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

    # Detection in single frame. Also saves the network probability map to the second ClickPoints layer
    # tif file of the probability map is saved to ClickPoints temporary folder.
    def detect_single(self):
        im = self.cp.getImage()
        img = self.cp.getImage().data
        frame = im.frame
        cells, probability_map = self.detect(im, img, frame)
        self.cp.reloadMask()
        self.cp.reloadMarker()

        # writing probability map as an additional layer
        filename = os.path.join(self.prob_folder, "%dprob_map.tiff" % frame)
        Image.fromarray((probability_map * 255).astype(np.uint8)).save(filename)
        # Catch error if image already exists. In this case only overwriting the image file is sufficient.
        try:
            self.db.setImage(filename=filename, sort_index=frame, layer=self.prob_layer, path=self.prob_path)
        except peewee.IntegrityError:
            pass

    # Base detection function. Includes filters for objects without fully closed boundaries, objects close to
    # the horizontal image edge and objects with a radius smaller the self.r_min.
    def detect(self, im, img, frame):

        if self.unet is None:
            self.unet = UNet((img.shape[0], img.shape[1], 1), 1, d=8)
        img = (img - np.mean(img)) / np.std(img).astype(np.float32)
        timestamp = getTimestamp(self.vidcap, frame)

        probability_map = self.unet.predict(img[None, :, :, None])[0, :, :, 0]
        prediction_mask = probability_map > 0.5
        cells, prediction_mask = mask_to_cells_edge(prediction_mask, img, self.config, self.rmin, {}, edge_dist=15,
                                                    return_mask=True)

        [c.update({"frames": frame, "timestamp": timestamp, "area": np.pi * (c["long_axis"] * c["short_axis"])/4}) for c in cells]  # maybe use map for this?

        self.db.setMask(image=im, data=prediction_mask.astype(np.uint8))
        self.db.deleteEllipses(type=self.marker_type_cell2, image=im)
        self.drawEllipse(pd.DataFrame(cells), self.marker_type_cell2)

        return cells, probability_map

    def keyPressEvent(self, event):

        if event.key() == QtCore.Qt.Key_G:
            print("detecting")
            self.detect_single()
            print("detecting finished")

    # Display all ellipses at launch
    def display_ellipses(self, type="cell", data=None):

        batch_size = 200
        data = data if not (data is None) else self.data_all_existing
        if len(data) == 0:
            return

        self.thread.thread_started.emit((0, len(data)), "displaying ellipses")
        for block in range(0, len(data), batch_size):
            if self.stop:
                break
            if block + batch_size > len(data):
                data_block = data.iloc[block:]
            else:
                data_block = data.iloc[block:block + batch_size]

            self.drawEllipse(data_block, type)
            self.thread.thread_progress.emit(block)
            self.cp.reloadMarker()  # not sure how thread safe this is
        self.thread.thread_finished.emit(len(data))

    # based ellipse display function
    def drawEllipse(self, data_block, type):

        if len(data_block) == 0:
            return

        strains = (data_block["long_axis"] - data_block["short_axis"]) / np.sqrt(
            data_block["long_axis"] * data_block["short_axis"])
        # list of all marker texts
        text = []
        for s, sol, irr in zip(strains, data_block['solidity'], data_block['irregularity']):
            text.append(f"strain {s:.3f}\nsolidity {sol:.2f}\nirreg. {irr:.3f}")
        self.db.setEllipses(frame=list(data_block["frames"]), x=list(data_block["x"]),
                            y=list(data_block["y"]), width=list(data_block["long_axis"] / self.config["pixel_size"]),
                            height=list(data_block["short_axis"] / self.config["pixel_size"]),
                            angle=list(data_block["angle"]), type=type, text=text)
