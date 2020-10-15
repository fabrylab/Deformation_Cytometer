import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scripts.helper_functions import getInputFile, getConfig, getData, getInputFolder
from scripts.helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from scripts.helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from scripts.helper_functions import storeEvaluationResults, plotDensityScatter, plotStressStrainFit, plotBinnedData, load_all_data, get_pressures
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import pylustrator

pylustrator.start()

""" THP1 """

fit_data = []

pressures = np.sort(np.unique(get_pressures(rf"\\131.188.117.96\biophysDS\meroles\2020.05.27_THP1_RPMI_2pc_Ag\THP1_27_05_2020_2replicate\2\*_result.txt")))
pressures = pressures[2:]
# iterate over all times
for index, pressure in enumerate(pressures):
    f = []
    time = 2
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\meroles\2020.05.27_THP1_RPMI_2pc_Ag\THP1_27_05_2020_2replicate\2\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data(path, pressure=pressure)
        f.append([config["fit"]["p"][0], config["fit"]["p"][1], config["fit"]["p"][0] * config["fit"]["p"][1]])

    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the three different plots
for i in range(3):
    plt.subplot(1, 3, i+1)
    print("i", np.mean(fit_data[:, :, i], axis=1))
    plt.errorbar(pressures, np.mean(fit_data[:, :, i], axis=1),
                 np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3,
                 label="THP1",
                 )


""" K562 """

fit_data = []
x = []
# iterate over all times
pressures = np.sort(np.unique(get_pressures(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\2\*_result.txt")))
for index, pressure in enumerate(pressures):
    f = []
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\2\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\2\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data(path, pressure=pressure)
        f.append([config["fit"]["p"][0], config["fit"]["p"][1], config["fit"]["p"][0] * config["fit"]["p"][1]])

    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the three different plots
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.errorbar(pressures, np.mean(fit_data[:, :, i], axis=1),
                 np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3,
                 label="K562",
                 )


""" NIH3T3 """

fit_data = []

pressures = np.sort(np.unique(get_pressures(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\2\*_result.txt")))
for index, pressure in enumerate(pressures):
    f = []
    # iterate over the different experiment paths
    for path in [
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\2\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data(path, pressure=pressure)
        f.append([config["fit"]["p"][0], config["fit"]["p"][1], config["fit"]["p"][0] * config["fit"]["p"][1]])

    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the three different plots
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.errorbar(pressures, np.mean(fit_data[:, :, i], axis=1),
                 np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3,
                 label="NIH 3T3",
                 )

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
        for index, file in enumerate(paths):#[repetition:repetition+1]):
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

        #plotDensityScatter(data.stress, data.strain)
        #plotStressStrainFit(data, config)
        #plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])
        #plt.title(f'{config["fit"]["p"][0] * config["fit"]["p"][1]:.2f}')
        fit_data.append([config["fit"]["p"][0], config["fit"]["p"][1], config["fit"]["p"][0] * config["fit"]["p"][1]])

        return fit_data


    #""" loading data """
    ## get the results file (by config parameter or user input dialog)

    datasets = [
    {
            "datafiles": [
                Path(r"Z:\meroles\2020.05.27_THP1_RPMI_2pc_Ag\THP1_27_05_2020_2replicate\2"),
            ],
            "rows": 2,
            "cols": 12,
            "range": np.arange(1, 7, 1),
            "type": "mar",
            "fitfunc": "log",
            "repetition": 2,
            "color": "C0",
            "cell": "THP1",
        },
        {
            "datafiles": [
                Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_july\07_07_2020_alginate2%_rpmi_no_fcs_time\2"),
                Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\2"),
            ],
            "rows": 2,
            "cols": 12,
            "type": "elham",
            "range": np.arange(2, 11, 1),
            "fitfunc": "line",
            "repetition": 2,
            "color": "C1",
            "cell": "K562",

        },
        {
            "datafiles": [
                Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_may\2020_05_22_alginateDMEM2%\2"),
            #    Path(r"Z:\emirzahossein\data\microscope4_baslercamera\2020_july\07_07_2020_alginate2%_rpmi_no_fcs_time"),
            ],
            "rows": 1,
            "cols": 12,
            "type": "elham",
            "range": np.arange(2, 14, 1),
            "fitfunc": "line",
            "repetition": 1,
            "color": "C2",
            "cell": "NIH 3T3",
    }
    ]

    for dataset in datasets:
        fit_data_all = []
        ax = None

        datafiles = dataset["datafiles"]

        for data_index, datafile in enumerate(datafiles):
            paths = []
            pressures = []
            #
            for index, file in enumerate(Path(datafile).glob("*_result.txt")):
                config = getConfig(file)
                paths.append(file)
                pressures.append(config['pressure_pa'] / 100_000)

            paths = np.array(paths)
            pressures = np.array(pressures)

            unique_pressures = np.unique(pressures)
            unique_pressures = unique_pressures[unique_pressures > 0.5]
            print(unique_pressures)

            fit_data = []
            index = 1
            for pressure in unique_pressures:
                f = plotPathList(paths[pressures==pressure])
                fit_data.append(f)
                index += 1
            fit_data_all.append(fit_data)

        fit_data_all = np.array(fit_data_all)

        def fit_func(x, p1):
            return p1*np.log(x+1)

        for i in range(3):
            plt.subplot(1, 3, i+1)
            fit_data = fit_data_all[:, :, 0, i]
            print(fit_data_all.shape, fit_data)
            from scipy.optimize import curve_fit

            if dataset["type"] == "mar":
                x = unique_pressures
            else:
                x = unique_pressures
            y = np.mean(fit_data, axis=0)

            plt.errorbar(x, np.mean(fit_data, axis=0), np.std(fit_data, axis=0)/np.sqrt(fit_data.shape[0]), capsize=3, color=dataset["color"], label=dataset["cell"])
            plt.plot(x, np.mean(fit_data, axis=0), "o", color=dataset["color"])
            plt.ylim(bottom=0)
            plt.xlim(left=0)

            x1 = np.arange(0, 4, 1)#np.linspace(plt.xlim()[0], plt.xlim()[1], 100)

            if 1:#dataset["fitfunc"] == "line":
                m, t = np.polyfit(x, y, deg=1)
                plt.plot(x1, m*x1+t, "--", color=dataset["color"])
            else:
                p, cov = curve_fit(fit_func, x, y, [1], bounds=[0, np.inf])
                plt.plot(x1, fit_func(x1, *p), "k")
            plt.xlabel("pressure (Pa)")
            plt.ylabel("stiffness (Pa)")

plt.ylim(top=130)
plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(15.980000/2.54, 5.970000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.062779, 0.202909, 0.241927, 0.689799])
plt.figure(1).axes[0].set_xlim(0.0, 3.2)
plt.figure(1).axes[0].set_xticklabels(["0", "1", "2", "3"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].set_xticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[0].set_ylim(0.0, 4.0)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.209737, 1.038176])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("alpha")
plt.figure(1).axes[1].set_position([0.416761, 0.202909, 0.226609, 0.710959])
plt.figure(1).axes[1].set_xlim(0.0, 3.2)
plt.figure(1).axes[1].set_xticklabels(["0", "1", "2", "3"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].set_xticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[1].set_ylim(0.0, 293.94600403095797)
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.350939, 1.007277])
plt.figure(1).axes[1].texts[0].set_text("b")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("prestress (Pa)")
plt.figure(1).axes[2].legend(handletextpad=0.7999999999999999, fontsize=8.0, title_fontsize=7.0)
plt.figure(1).axes[2].set_position([0.755425, 0.202909, 0.226609, 0.710959])
plt.figure(1).axes[2].set_xlim(0.0, 3.2)
plt.figure(1).axes[2].set_xticklabels(["0", "1", "2", "3"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[2].set_xticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[2].set_ylim(0.0, 320.0)
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].get_legend()._set_loc((0.453021, 0.771336))
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_position([-0.343916, 1.007277])
plt.figure(1).axes[2].texts[0].set_text("c")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).axes[2].get_yaxis().get_label().set_text("stiffness (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()


