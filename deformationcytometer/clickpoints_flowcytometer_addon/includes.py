from deformationcytometer.includes.includes import Dialog
import deformationcytometer.detection.includes.UNETmodel
from pathlib import Path
from qtpy import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal
from collections import defaultdict
import json
import numpy as np
import pandas as pd
from deformationcytometer.includes.includes import getData, getConfig
from deformationcytometer.evaluation.helper_functions import  get_folders, check_config_changes,getVelocity,\
    correctCenter, filterCells, getStressStrain, apply_velocity_fit, get_cell_properties

def doNothing(x):
    # this is the most important function in the history of functions ... ever!
    return x


tooltip_strings = defaultdict(str)
tooltip_strings["choose file"] = "Select a neural network weights file (.h5)"
tooltip_strings["single detection"] = "Update the cell detection in a singel frame. This will display the detection " \
                                      "mask, the elliptical fit (cell new) and the prediction probability map " \
                                      "(in a new ClickPoints layer, accessible with the 'page up' key)." \
                                      " Keyboard shortcut: press g"
tooltip_strings["update cell detection"] = "Update the cell detection in a range of frames specified " \
                                           "by the ClickPoints sliders in the ClickPoints timeline."
# TODO improve formulation
tooltip_strings["irregularity"] = "Set the irregularity threshold used to filter data generated " \
                                  "with 'update cell detection'."
tooltip_strings["solidity"] = "Set the solidity threshold used to filter data generated with 'update cell detection'."
tooltip_strings["stress-strain"] = "Display a cell stress vs cell strain scatter plot."
tooltip_strings["regularity-solidity"] = "Display a cell irregularity vs cell solidity scatter plot."
tooltip_strings["k histogram"] = "Display a histogram of cell stiffness (k)."
tooltip_strings["alpha histogram"] = "Display a histogram of cell viscosity (alpha)."
# TODO: stiffness and viscosity correct?
tooltip_strings["k-alpha"] = "Display a cell stiffness (k) vs cell strain scatter plot."
tooltip_strings["displaying existing data"] = "Click to switch between displaying data loaded from" \
                                              "existing result files and newly generated data. You also " \
                                              "need to press the plotting button again."
tooltip_strings["stop"] = "Terminate the currently running process."
tooltip_strings["min radius"] = "Threshold for minimal cell size. This threshold is not applied to the prediction mask."


default_config_path = Path(deformationcytometer.detection.includes.UNETmodel.__file__).parent
default_config_path = default_config_path.joinpath("default_config.txt")

# Starting process (displaying the ellipses and cell detection for multiple frames) in a separate process
class Worker(QtCore.QThread):
    # Signals are used to communicate with the main qt window. Specifically the progress bar.
    # Directly manipulating the progress bar from the run function is not thread safe (two processes may try
    # to change the progress bar at the same time)
    thread_started = QtCore.Signal(tuple, str)
    thread_finished = QtCore.Signal(int)
    thread_progress = QtCore.Signal(int)

    def __init__(self, parent=None, run_function=None):
        QtCore.QThread.__init__(self, parent)
        self.run_function = run_function

    def run(self):
        self.run_function()

# layout for file selection
class SetFile(QtWidgets.QHBoxLayout):
    fileSeleted = pyqtSignal(bool)

    def __init__(self, file=None, type="file", filetype=""):
        super().__init__()  # activating QVboxLayout
        if file is None:
            self.file = ""
        else:
            self.file = file
        self.filetype = filetype
        self.type = type

        # line edit holding the currently selected folder
        self.line_edit_folder = QtWidgets.QLineEdit(str(self.file))
        self.line_edit_folder.editingFinished.connect(self.emitTextChanged)
        self.addWidget(self.line_edit_folder, stretch=4)

        # button to browse folders
        self.open_folder_button = QtWidgets.QPushButton("choose file")
        self.open_folder_button.setToolTip(tooltip_strings["choose file"])
        self.open_folder_button.clicked.connect(self.file_dialog)
        self.addWidget(self.open_folder_button, stretch=2)

    def file_dialog(self):
        dialog = Dialog(title="open file", filetype=self.filetype, mode="file",
                        settings_name="Deformationcytometer Addon")
        self.file = dialog.openFile()
        self.fileSeleted.emit(True)
        self.line_edit_folder.setText(self.file)


    def emitTextChanged(self):
        self.fileSeleted.emit(True)


#TODo: update to new analysis pipe line
def load_all_data_old(input_path, solidity_threshold=0.96, irregularity_threshold=1.06, pressure=None, repetition=None, new_eval=False):
    global ax

    evaluation_version = 8

    paths = get_folders(input_path, pressure=pressure, repetition=repetition)
    fit_data = []
    data_list = []
    filters = []
    config = {}
    for index, file in enumerate(paths):
        #print(file)
        output_file = Path(str(file).replace("_result.txt", "_evaluated.csv"))
        output_config_file = Path(str(file).replace("_result.txt", "_evaluated_config.txt"))

        # load the data and the config
        data = getData(file)
        config = getConfig(file)
        config["channel_width_m"] = 0.00019001261833616293

        if output_config_file.exists():
            with output_config_file.open("r") as fp:
                config = json.load(fp)
                config["channel_width_m"] = 0.00019001261833616293

        config_changes = check_config_changes(config, evaluation_version, solidity_threshold, irregularity_threshold)
        if "filter" in config:
                filters.append(config["filter"])


        """ evaluating data"""
        if not output_file.exists() or config_changes or new_eval:

            getVelocity(data, config)
            # take the mean of all values of each cell
            data = data.groupby(['cell_id'], as_index=False).mean()

            tt_file = Path(str(file).replace("_result.txt", "_tt.csv"))
            if tt_file.exists():
                data.set_index("cell_id", inplace=True)
                data_tt = pd.read_csv(tt_file)
                data["omega"] = np.zeros(len(data)) * np.nan
                for i, d in data_tt.iterrows():
                    if d.tt_r2 > 0.2:
                        data.at[d.id, "omega"] = d.tt * 2 * np.pi

                data.reset_index(inplace=True)
            else:
                print("WARNING: tank treading has not been evaluated yet")

            correctCenter(data, config)

            data = filterCells(data, config, solidity_threshold, irregularity_threshold)
            # reset the indices
            data.reset_index(drop=True, inplace=True)

            getStressStrain(data, config)

            #data = data[(data.stress < 50)]
            data.reset_index(drop=True, inplace=True)

            data["area"] = data.long_axis * data.short_axis * np.pi
            data["pressure"] = config["pressure_pa"]*1e-5

            data, p = apply_velocity_fit(data)


            omega, mu1, eta1, k_cell, alpha_cell, epsilon = get_cell_properties(data)

            try:
                config["evaluation_version"] = evaluation_version
                config["network_evaluation_done"] = True
                config["solidity"] = solidity_threshold
                config["irregularity"] = irregularity_threshold
                data.to_csv(output_file, index=False)
                #print("config", config, type(config))
                with output_config_file.open("w") as fp:
                    json.dump(config, fp, indent=0)

            except PermissionError:
                pass

        else:
            with output_config_file.open("r") as fp:
                config = json.load(fp)
                config["channel_width_m"] = 0.00019001261833616293

        data = pd.read_csv(output_file)

        #data = data[(data.area > 0) * (data.area < 2000) * (data.stress < 250)]
        #data.reset_index(drop=True, inplace=True)

        data_list.append(data)
    l_before = np.sum([d["l_before"] for d in filters])
    l_after = np.sum([d["l_after"] for d in filters])

    config["filter"] = {"l_before":l_before, "l_after":l_after}
    try:
        data = pd.concat(data_list)
    except ValueError:
        raise ValueError("No object found", input_path)
    data.reset_index(drop=True, inplace=True)

    #fitStiffness(data, config)
    return data, config