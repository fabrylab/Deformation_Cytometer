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
from helper_functions import getInputFile, getConfig, getData, getInputFolder
from helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from helper_functions import storeEvaluationResults, plotDensityScatter, plotStressStrainFit, plotBinnedData
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import pylustrator
pylustrator.start()
ax = None

xlimits = [-40, 300]
ylimits = [-0.1, 1.1]

def plotPathList(paths, repetition):
    global ax
    paths = list(paths)
    print(paths)
    fit_data = []

    data_list = []
    for index, file in enumerate(paths[repetition:repetition+1]):
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

    plotDensityScatter(data.stress, data.strain)
    plotStressStrainFit(data, config)
    #plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])
    #plt.title(f'{config["fit"]["p"][0] * config["fit"]["p"][1]:.2f}')
    fit_data.append(config["fit"]["p"][0] * config["fit"]["p"][1])

    return fit_data


#""" loading data """
## get the results file (by config parameter or user input dialog)
datasets = [
    {
        "datafiles": [
            Path(r"\\131.188.117.96\biophysDS\meroles\2020.07.07.Control.sync\Control2"),
            Path(r"\\131.188.117.96\biophysDS\meroles\2020.06.26\Control2"),
            Path(r"\\131.188.117.96\biophysDS\meroles\2020.06.26\Control1"),
            Path(r"\\131.188.117.96\biophysDS\meroles\2020.07.09_control_sync_hepes\Control3")
        ],
        "rows": 2,
        "cols": 12,
        "range": range(1, 6, 1),
        "fitfunc": "log",
        "repetition": 2,
    },
    {
        "datafiles": [
            Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_july\2020_07_10_alginate2%_K562_0%FCS_time"),
            Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_july\07_07_2020_alginate2%_rpmi_no_fcs_time"),
        ],
        "rows": 2,
        "cols": 12,
        "range": range(2, 11, 2),
        "fitfunc": "line",
        "repetition": 2,
    },
    {
        "datafiles": [
            Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_may\2020_05_22_alginateDMEM2%"),
        #    Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_july\07_07_2020_alginate2%_rpmi_no_fcs_time"),
        ],
        "rows": 1,
        "cols": 12,
        "range": range(2, 11, 2),
        "fitfunc": "line",
        "repetition": 1,
}
]

rows = 3
cols = 5
#row_index = 0
data_index = -1
for dataset in datasets:
    data_index += 1
    fit_data_all = []
    ax = None

    datafiles = dataset["datafiles"]

    #for data_index, datafile in enumerate(datafiles):
    for datafile in datafiles[:1]:
        fit_data = []
        index = 1
        for time in dataset["range"]:
            if ax is None:
                ax = plt.subplot(rows, cols, index + cols*data_index)
            else:
                ax = plt.subplot(rows, cols, index + cols*data_index)#, sharex=ax, sharey=ax)

            if index != 1:
                plt.tick_params(labelleft='off')
            else:
                plt.ylabel("strain")

            if data_index < rows-1:
                plt.tick_params(labelbottom='off')
            else:
                plt.xlabel("stress (pa)")

            if dataset["fitfunc"] != "line":
                f = plotPathList((datafile / f"T{time*10}").glob("*_result.txt"), repetition=dataset["repetition"])
            else:
                f = plotPathList((datafile / f"{time}").glob("*_result.txt"), repetition=dataset["repetition"])
            fit_data.append(f)
            index += 1
        fit_data_all.append(fit_data)

    xmin = np.min([ax.get_xlim()[0] for ax in plt.gcf().axes])
    xmax = np.max([ax.get_xlim()[1] for ax in plt.gcf().axes])
    ymin = np.min([ax.get_ylim()[0] for ax in plt.gcf().axes])
    ymax = np.max([ax.get_ylim()[1] for ax in plt.gcf().axes])
    for ax in plt.gcf().axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    fit_data_all = np.array(fit_data_all)

#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_position([0.110316, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[0].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[0].set_xticklabels(["", ""])
plt.figure(1).axes[0].set_xticks([0.0, 100.0])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.509148, 0.891045])
plt.figure(1).axes[0].texts[0].set_text("10 min")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("THP1\nstrain")
plt.figure(1).axes[1].set_position([0.289717, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[1].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[1].set_xticklabels(["", ""])
plt.figure(1).axes[1].set_xticks([0.0, 100.0])
plt.figure(1).axes[1].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[1].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[1].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.545742, 0.891045])
plt.figure(1).axes[1].texts[0].set_text("20 min")
plt.figure(1).axes[2].set_position([0.469119, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[2].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[2].set_xticklabels(["", ""])
plt.figure(1).axes[2].set_xticks([0.0, 100.0])
plt.figure(1).axes[2].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[2].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[2].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.504574, 0.891045])
plt.figure(1).axes[2].texts[0].set_text("30 min")
plt.figure(1).axes[3].set_position([0.648521, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[3].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[3].set_xticklabels(["", ""])
plt.figure(1).axes[3].set_xticks([0.0, 100.0])
plt.figure(1).axes[3].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[3].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[3].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_ha("center")
plt.figure(1).axes[3].texts[0].set_position([0.518297, 0.891045])
plt.figure(1).axes[3].texts[0].set_text("50 min")
plt.figure(1).axes[4].set_position([0.827922, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[4].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[4].set_xticklabels(["", ""])
plt.figure(1).axes[4].set_xticks([0.0, 100.0])
plt.figure(1).axes[4].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[4].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[4].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_ha("center")
plt.figure(1).axes[4].texts[0].set_position([0.532020, 0.891045])
plt.figure(1).axes[4].texts[0].set_text("60 min")
plt.figure(1).axes[5].set_position([0.110316, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[5].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[5].set_xticklabels(["", ""])
plt.figure(1).axes[5].set_xticks([0.0, 100.0])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].get_yaxis().get_label().set_text("K562\nstrain")
plt.figure(1).axes[6].set_position([0.289717, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[6].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[6].set_xticklabels(["", ""])
plt.figure(1).axes[6].set_xticks([0.0, 100.0])
plt.figure(1).axes[6].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[6].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[6].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[6].spines['right'].set_visible(False)
plt.figure(1).axes[6].spines['top'].set_visible(False)
plt.figure(1).axes[7].set_position([0.469119, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[7].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[7].set_xticklabels(["", ""])
plt.figure(1).axes[7].set_xticks([0.0, 100.0])
plt.figure(1).axes[7].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[7].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[7].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[7].spines['right'].set_visible(False)
plt.figure(1).axes[7].spines['top'].set_visible(False)
plt.figure(1).axes[8].set_position([0.648521, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[8].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[8].set_xticklabels(["", ""])
plt.figure(1).axes[8].set_xticks([0.0, 100.0])
plt.figure(1).axes[8].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[8].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[8].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[8].spines['right'].set_visible(False)
plt.figure(1).axes[8].spines['top'].set_visible(False)
plt.figure(1).axes[9].set_position([0.827922, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[9].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[9].set_xticklabels(["", ""])
plt.figure(1).axes[9].set_xticks([0.0, 100.0])
plt.figure(1).axes[9].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[9].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[9].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[9].spines['right'].set_visible(False)
plt.figure(1).axes[9].spines['top'].set_visible(False)
plt.figure(1).axes[10].set_position([0.110316, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[10].spines['right'].set_visible(False)
plt.figure(1).axes[10].spines['top'].set_visible(False)
plt.figure(1).axes[10].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[10].get_yaxis().get_label().set_text("NIH 3T3\nstrain")
plt.figure(1).axes[11].set_position([0.289717, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[11].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[11].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[11].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[11].spines['right'].set_visible(False)
plt.figure(1).axes[11].spines['top'].set_visible(False)
plt.figure(1).axes[11].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[12].set_position([0.469119, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[12].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[12].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[12].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[12].spines['right'].set_visible(False)
plt.figure(1).axes[12].spines['top'].set_visible(False)
plt.figure(1).axes[12].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[13].set_position([0.648521, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[13].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[13].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[13].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[13].spines['right'].set_visible(False)
plt.figure(1).axes[13].spines['top'].set_visible(False)
plt.figure(1).axes[13].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[14].set_position([0.827922, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[14].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[14].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[14].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[14].spines['right'].set_visible(False)
plt.figure(1).axes[14].spines['top'].set_visible(False)
plt.figure(1).axes[14].get_xaxis().get_label().set_text("stress (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()


