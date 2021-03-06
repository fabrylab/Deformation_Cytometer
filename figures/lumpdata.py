# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:07:32 2020

@author: user
"""

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
from deformationcytometer.includes.includes import getInputFile, getConfig, getData, getInputFolder
from deformationcytometer.evaluation.helper_functions import getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from deformationcytometer.evaluation.helper_functions import storeEvaluationResults, plotDensityScatter, plotStressStrainFit, plotBinnedData
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

        #data = data[(data.area > 0) * (data.area < 2000) * (data.stress < 250)]
        #data.reset_index(drop=True, inplace=True)

        data_list.append(data)


    data = pd.concat(data_list)
    data.reset_index(drop=True, inplace=True)

    fitStiffness(data, config)

    plotDensityScatter(data.stress, data.strain)
    #plotStressStrainFit(data, config)
    plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])
    #plt.title(f'{config["fit"]["p"][0] * config["fit"]["p"][1]:.2f}')
    fit_data.append(config["fit"]["p"][0] * config["fit"]["p"][1])

    return fit_data


#""" loading data """
## get the results file (by config parameter or user input dialog)
datasets = [
    {
        "datafiles": [
            #Path(r"Z:\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_24_alginate2.5%_dmem_NIH-3T3"),
        #    Path(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020"),
           Path(r"Z:\emirzahossein\microfluidic cell rhemeter data\microscope_1\october_2020\2020_10_07_thea\3T3"),
        ],
        "rows": 1,
        "cols": 3,
        "range": range(1, 3, 1),
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

        f = plotPathList(paths[pressures==pressure])
        fit_data.append(f)
        index += 1

    xmin = np.min([ax.get_xlim()[0] for ax in plt.gcf().axes])
    xmax = np.max([ax.get_xlim()[1] for ax in plt.gcf().axes])
    ymin = np.min([ax.get_ylim()[0] for ax in plt.gcf().axes])
    ymax = np.max([ax.get_ylim()[1] for ax in plt.gcf().axes])
    for ax in plt.gcf().axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 4.390000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.387173, 0.263334, 0.268860, 0.717020])
plt.figure(1).axes[0].set_xlim(0.0, 90.0)
#plt.figure(1).axes[0].set_xticklabels(["-75", "0", "75"])
plt.figure(1).axes[0].set_xticks([0.0, 75.0])
plt.figure(1).axes[0].set_xticks([np.nan], minor=True)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.509148, 0.891045])
plt.figure(1).axes[0].texts[0].set_text("3 bar")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_position([-0.232258, 0.955913])
plt.figure(1).axes[0].texts[1].set_text("a")
plt.figure(1).axes[0].texts[1].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("stress (pa)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("NIH 3T3\nstrain")
#% end: automatic generated code from pylustrator
#plt.savefig(__file__[:-3]+".png", dpi=300)
#plt.savefig(__file__[:-3]+".pdf")
plt.show()