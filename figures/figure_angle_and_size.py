# -*- coding: utf-8 -*-
"""
Created on Tue May 22 2020

@author: Ben

# This program reads a txt file with the analyzed cell position, shape (semi-major and semi-minor axis etc.),
# computes the cell strain and the fluid shear stress acting on each cell,
# plots the data (strain versus stress) for each cell using a kernel density estimate for the datapoint color,
# and fits a stress stiffening equation to the data 
# The results such as maximum flow speed, cell mechanical parameters, etc. are stored in 
# the file 'all_data.txt' located at the same directory as this script 
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scripts.helper_functions import getInputFile, getConfig, getData, getInputFolder
from scripts.helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from scripts.helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from scripts.helper_functions import storeEvaluationResults, plotDensityScatter, plotStressStrainFit, plotBinnedData, load_all_data
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import pylustrator
pylustrator.start()

for index, pressure in enumerate([1]):
    ax = plt.subplot(1, 3, index+1)

    #data, config = load_all_data(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data"+
    #                             r"\microscope4\2020_may\2020_05_22_alginateDMEM2%\[0-9]\*_result.txt", pressure=pressure)
    data, config = load_all_data([
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\[0-9]\*_result.txt",
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\[0-9]\*_result.txt",
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_27_alginate2%_dmem_NIH_time_1\[0-9]\*_result.txt",
    ], pressure=pressure)

    plt.subplot(1, 2, 1)
    plotDensityScatter(data.rp, data.angle)

    plt.subplot(1, 2, 2)
    plotDensityScatter(data.rp, np.sqrt(data.area / np.pi))

if 0:
    ax = None

    xlimits = [-40, 300]
    ylimits = [-0.1, 1.1]

    def plotPathList(paths):
        global ax
        paths = list(paths)
        print(paths)
        fit_data = []

        data_list = []
        for index, file in enumerate(paths):
            output_file = Path(str(file).replace("_result.txt", "_evaluated.csv"))

            # load the data and the config
            data = getData(file)
            config = getConfig(file)

            """ evaluating data"""
            if not output_file.exists():
                #refetchTimestamps(data, config)

                getVelocity(data, config)

                # take the mean of all values of each cell
                data = data.groupby(['cell_id']).mean()

                correctCenter(data, config)

                data = filterCells(data, config)

                # reset the indices
                data.reset_index(drop=True, inplace=True)

                getStressStrain(data, config)

                #data = data[(data.stress < 50)]
                data.reset_index(drop=True, inplace=True)

                data["area"] = data.long_axis * data.short_axis * np.pi
                data.to_csv(output_file, index=False)

            data = pd.read_csv(output_file)

            #data = data[(data.area > 0) * (data.area < 2000) * (data.stress < 250)]
            #data.reset_index(drop=True, inplace=True)

            data_list.append(data)


        data = pd.concat(data_list)
        data.reset_index(drop=True, inplace=True)

        fitStiffness(data, config)

        plotDensityScatter(data.rp, data.angle)
        #plotStressStrainFit(data, config)
        #plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])
        #plt.title(f'{config["fit"]["p"][0] * config["fit"]["p"][1]:.2f}')
        fit_data.append(config["fit"]["p"][0] * config["fit"]["p"][1])

        return fit_data

    def plotPathList2(paths):
        global ax
        paths = list(paths)
        print(paths)
        fit_data = []

        data_list = []
        for index, file in enumerate(paths):
            output_file = Path(str(file).replace("_result.txt", "_evaluated.csv"))

            # load the data and the config
            data = getData(file)
            config = getConfig(file)

            """ evaluating data"""
            if not output_file.exists():
                #refetchTimestamps(data, config)

                getVelocity(data, config)

                # take the mean of all values of each cell
                data = data.groupby(['cell_id']).mean()

                correctCenter(data, config)

                data = filterCells(data, config)

                # reset the indices
                data.reset_index(drop=True, inplace=True)

                getStressStrain(data, config)

                #data = data[(data.stress < 50)]
                data.reset_index(drop=True, inplace=True)

                data["area"] = data.long_axis * data.short_axis * np.pi
                data.to_csv(output_file, index=False)

            data = pd.read_csv(output_file)

            #data = data[(data.area > 0) * (data.area < 2000) * (data.stress < 250)]
            #data.reset_index(drop=True, inplace=True)

            data_list.append(data)


        data = pd.concat(data_list)
        data.reset_index(drop=True, inplace=True)

        fitStiffness(data, config)

        plotDensityScatter(data.rp, np.sqrt(data.area/np.pi))
        #plotStressStrainFit(data, config)
        #plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])
        #plt.title(f'{config["fit"]["p"][0] * config["fit"]["p"][1]:.2f}')
        fit_data.append(config["fit"]["p"][0] * config["fit"]["p"][1])

        return fit_data

    #""" loading data """
    ## get the results file (by config parameter or user input dialog)
    datasets = [
        {
            "datafiles": [
                Path(r"\\131.188.117.96\biophysDS\emirzahossein\data\microscope4_baslercamera\2020_may\2020_05_22_alginateDMEM2%"),
            #    Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_july\07_07_2020_alginate2%_rpmi_no_fcs_time"),
            ],
            "rows": 1,
            "cols": 12,
            "range": range(2, 11, 2),
            "fitfunc": "line",
            "repetition": 1,
    }
    ]

    rows = 1
    cols = 3
    #row_index = 0
    data_index = -1
    dataset = datasets[0]
    datafiles = dataset["datafiles"]
    for data_index, datafile in enumerate(datafiles):
        data_index += 1
        paths = []
        pressures = []
        ax = None
        datafiles = dataset["datafiles"]
        #
        for index, file in enumerate(Path(datafile).glob("**/*_result.txt")):
            config = getConfig(file)
            paths.append(file)
            pressures.append(config['pressure_pa'] / 100_000)

        paths = np.array(paths)
        pressures = np.array(pressures)

        unique_pressures = np.unique(pressures)
        unique_pressures = [1, 2, 3]#unique_pressures[unique_pressures > 0.5]
        print(unique_pressures)

        fit_data = []
        index = 1

        plt.subplot(1, 2, 1)
        f = plotPathList(paths[pressures == 1])

        plt.subplot(1, 2, 2)
        f = plotPathList2(paths[pressures == 1])


#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.000000/2.54, 4.370000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.118367, 0.241321, 0.356764, 0.716233])
plt.figure(1).axes[0].set_xlim(-90.0, 90.0)
plt.figure(1).axes[0].set_xticklabels(["-75", "0", "75"])
plt.figure(1).axes[0].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[0].set_xticks([np.nan], minor=True)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[0].set_position([-0.232258, 0.955913])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("angle (deg)")
plt.figure(1).axes[1].set_position([0.618713, 0.241321, 0.356764, 0.716233])
plt.figure(1).axes[1].set_xlim(-90.0, 90.0)
plt.figure(1).axes[1].set_xticklabels(["-75", "0", "75"])
plt.figure(1).axes[1].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[1].set_xticks([np.nan], minor=True)
plt.figure(1).axes[1].set_ylim(0.0, 30.0)
plt.figure(1).axes[1].set_yticklabels(["0", "5", "10", "15", "20", "25", "30"])
plt.figure(1).axes[1].set_yticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].xaxis.labelpad = 3.716691
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[1].new
plt.figure(1).axes[1].texts[0].set_position([-0.354809, 0.955913])
plt.figure(1).axes[1].texts[0].set_text("b")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("undeformed\ncell diameter (µm)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


