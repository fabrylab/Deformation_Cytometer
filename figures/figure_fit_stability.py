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
from scripts.helper_functions import storeEvaluationResults, plotDensityScatter, plotStressStrainFit, plotBinnedData
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import pylustrator
pylustrator.start()
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

        if 0:
            plt.plot(data.rp, data.angle, "o")
            plt.axhline(0)
            plt.axvline(0)
            plt.axhline(45)
            plt.axhline(-45)
            print(data.angle)
            plt.show()


        #data = data[(data.area > 0) * (data.area < 2000) * (data.stress < 250)]
        #data.reset_index(drop=True, inplace=True)

        data_list.append(data)


    data = pd.concat(data_list)
    data.reset_index(drop=True, inplace=True)

    getStressStrain(data, config)


    if 0:
        if 1:
            data.strain[(data.angle > 0) & (data.rp > 0)] *= -1
            data.strain[(data.angle < 0) & (data.rp < 0)] *= -1
        else:
            data.strain[(data.angle > 45)] *= -1
            data.strain[(data.angle < -45)] *= -1

    fits = []
    errors = []

    for i in np.arange(30, 250, 10):
        data2 = data[data.stress < i].reset_index(drop=True)
        print(i, len(data2))

        fitStiffness(data2, config)
        fits.append(config["fit"]["p"])
        errors.append(config["fit"]["err"])
        print("err", config["fit"]["err"], errors)

    plotDensityScatter(data.stress, data.strain)
    plotStressStrainFit(data, config)
    plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])
    #plt.title(f'{config["fit"]["p"][0] * config["fit"]["p"][1]:.2f}')
    fit_data.append(config["fit"]["p"][0] * config["fit"]["p"][1])

    return fits, errors#fit_data


#""" loading data """
## get the results file (by config parameter or user input dialog)
datasets = [
    {
        "datafiles": [
            Path(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4_baslercamera\2020_may\2020_05_22_alginateDMEM2%"),
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
cols = 4
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
    unique_pressures = [3]#unique_pressures[unique_pressures > 0.5]
    print(unique_pressures)

    fit_data = []
    index = 1

    #for data_index, datafile in enumerate(datafiles):
    fit_data = []
    index = 1
    for pressure in unique_pressures:
        print(rows, cols, index, data_index)
        if ax is None:
            ax = plt.subplot(rows, cols, index)
        else:
            ax = plt.subplot(rows, cols, index)#, sharex=ax, sharey=ax)

        if index != 1:
            plt.tick_params(labelleft='off')
        else:
            plt.ylabel("strain")

        if data_index < rows-1:
            plt.tick_params(labelbottom='off')
        else:
            plt.xlabel("stress (pa)")

        f, errors = plotPathList(paths[pressures==pressure])
        fit_data.append(f)
        index += 1

    x = np.arange(30, 250, 10)
    fit_data = np.array(fit_data)[0]
    errors = np.array(errors)
    print("fit_data", fit_data.shape, errors.shape)
    plt.subplot(1, 4, 2)
    plt.fill_between(x, fit_data[:, 0]+errors[:, 0], fit_data[:, 0]-errors[:, 0], color="gray")
    plt.plot(x, fit_data[:, 0])
    plt.subplot(1, 4, 3)
    plt.fill_between(x, fit_data[:, 1]+errors[:, 1], fit_data[:, 1]-errors[:, 1], color="gray")
    plt.plot(x, fit_data[:, 1])

    plt.subplot(1, 4, 4)
    plt.fill_between(x, fit_data[:, 0]*fit_data[:, 1]+errors[:, 2], fit_data[:, 0]*fit_data[:, 1]-errors[:, 2], color="gray")
    plt.plot(x, fit_data[:, 0]*fit_data[:, 1])

#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 4.390000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.078263, 0.240487, 0.239173, 0.717020])
plt.figure(1).axes[0].set_xlim(-10.0, 270.0)
plt.figure(1).axes[0].set_xticklabels(["0", "100", "200"])
plt.figure(1).axes[0].set_xticks([0.0, 100.0, 200.0])
plt.figure(1).axes[0].set_xticks([np.nan], minor=True)
plt.figure(1).axes[0].set_ylim(0.0, 1.2)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_xaxis().get_label().set_text("shear stress (Pa)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("strain")
plt.figure(1).axes[1].set_position([0.410016, 0.240487, 0.168478, 0.717020])
plt.figure(1).axes[1].set_xlim(-10.0, 270.0)
plt.figure(1).axes[1].set_xticks([np.nan], minor=True)
plt.figure(1).axes[1].set_ylim(0.0, 5.0)
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].xaxis.labelpad = 3.716691
plt.figure(1).axes[1].get_xaxis().get_label().set_text("maximum shear stress (Pa)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("stiffening factor")
plt.figure(1).axes[2].set_position([0.602707, 0.240487, 0.168478, 0.717020])
plt.figure(1).axes[2].set_xlim(-10.0, 270.0)
plt.figure(1).axes[2].set_xticks([np.nan], minor=True)
plt.figure(1).axes[2].set_ylim(0.0, 100.0)
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].yaxis.labelpad = -4.720000
plt.figure(1).axes[2].get_xaxis().get_label().set_text("maximum shear stress (Pa)")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("prestress (Pa)")
plt.figure(1).axes[3].set_position([0.814025, 0.240487, 0.168478, 0.717020])
plt.figure(1).axes[3].set_ylim(0.0, 100.0)
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


