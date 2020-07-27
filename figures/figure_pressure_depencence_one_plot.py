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

global_im = None
global_index = 0

def plotPathList(paths, cmap=None, alpha=None):
    global ax, global_im
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

    #plotDensityScatter(data.stress, data.strain, cmap=cmap, alpha=0.5)
    def densityPlot(x, y, cmap, alpha=0.5):
        global global_im, global_index
        from scipy.stats import kde

        ax = plt.gca()

        # Thus we can cut the plotting window in several hexbins
        nbins = np.max(x)/10
        ybins = 20

        # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
        k = kde.gaussian_kde(np.vstack([x, y]))
        if 0:
            xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():ybins * 1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            # plot a density
            ax.set_title('Calculate Gaussian KDE')
            ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', alpha=alpha, cmap=cmap)
        else:
            xi, yi = np.meshgrid(np.linspace(-10, 250, 200), np.linspace(0, 1, 80))#np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():ybins * 1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            im = zi.reshape(xi.shape)
            if 0:
                if global_im is None:
                    global_im = np.zeros((im.shape[0], im.shape[1], 3), dtype="uint8")
                if 1:#global_index == 1:
                    print("_____", im.min(), im.max())
                    im -= np.percentile(im, 10)
                    global_im[:, :, global_index] = im/im.max()*255
                    print("_____", global_im[:, :, global_index].min(), global_im[:, :, global_index].max())
                print("COLOR", global_index)
                global_index += 1
                if global_index == 3:
                    print(global_im.shape, global_im.dtype)
                    plt.imshow(global_im[::-1], extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)], aspect="auto")
            else:
                if global_im is None:
                    global_im = []
                im -= im.min()
                im /= im.max()
                global_im.append(plt.get_cmap(cmap)(im**0.5))
                global_im[-1][:, :, 3] = im
                plt.imshow(global_im[-1][::-1], vmin=0, vmax=1, extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)],
                           aspect="auto")
                global_index += 1
                if global_index == 3:
                    print("COLOR", global_im[0].shape, global_im[0].min(), global_im[0].max())
                    im = global_im[0] + global_im[1] + global_im[2] - 2
                    #im[im<0] = 0
                    #im[im>255] = 255
                    print("COLOR", im.shape, im.min(), im.max())
                    #plt.imshow(im[::-1], vmin=0, vmax=1, extent=[np.min(xi), np.max(xi), np.min(yi), np.max(yi)], aspect="auto")
    densityPlot(data.stress, data.strain, cmap=cmap, alpha=alpha)

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

    #for data_index, datafile in enumerate(datafiles):
    fit_data = []
    index = 1
    cmaps = ["", "Greens", "Blues", "Purples"]
    for pressure in unique_pressures[::-1]:
        print(rows, cols, index, data_index)

        f = plotPathList(paths[pressures==pressure], cmaps[index], 0.5 if index > 1 else 1)
        fit_data.append(f)
        index += 1
    for i in range(1, 4):
        plt.plot([], [], "o", color=plt.get_cmap(cmaps[::-1][i-1])(0.75), label=f"{i} bar")


plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(8.000000/2.54, 4.360000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.198362, 0.246488, 0.451831, 0.717020])
plt.figure(1).axes[0].set_xlim(-10.0, 270.0)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend()._set_loc((0.983850, 0.458041))
plt.figure(1).axes[0].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("strain")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


