import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from deformationcytometer.evaluation.helper_functions import getConfig, getData
from deformationcytometer.evaluation.helper_functions import getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from deformationcytometer.evaluation.helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from deformationcytometer.evaluation.helper_functions import storeEvaluationResults, plotDensityScatter, plotStressStrainFit, plotBinnedData, load_all_data, get_pressures
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import pylustrator

pylustrator.start()

""" THP1 """

# remove TPH1 data for now
if 0:
    fit_data = []
    x = []
    # iterate over all times
    #pressures = np.sort(np.unique(get_pressures(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\october_2020\2020_10_16_alginate2%_THP1_overtime\2\*_result.txt")))
    pressures = np.sort(np.unique(get_pressures(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\Mar THP1\2\*_result.txt")))
    pressures = pressures[2:]
    print(pressures)
    for index, pressure in enumerate(pressures):
        f = []
        # iterate over the different experiment paths
        for path in [
    #        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\october_2020\2020_10_15_alginate2%_THP1_overtime\**\*_result.txt",
            rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\Mar THP1\2\*_result.txt",
    #        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\Mar THP1\Control1\**\*_result.txt",
    #        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\Mar THP1\Control2\**\*_result.txt",
        ]:
            # get the data and the fit parameters
            data, config = load_all_data(path, pressure=pressure)
            f.append([np.mean(data.alpha_cell), np.nanmedian(data.k_cell)])
            #f.append([np.mean(data.alpha_cell), np.mean(np.log10(data.k_cell))])


        fit_data.append(f)

    fit_data = np.array(fit_data)

    # plot the fit data in the two different plots
    for i in range(2):
        plt.subplot(1, 2, i+1)
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
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\2\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\2\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_k562_0_5%_FCS_time\2\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data(path, pressure=pressure)
        f.append([np.mean(data.alpha_cell), np.nanmedian(data.k_cell)])
        #f.append([np.mean(data.alpha_cell), np.mean(np.log10(data.k_cell))])

    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the two different plots
for i in range(2):
    plt.subplot(1, 2, i+1)
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
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\2\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\2\*_result.txt",
    ]:
        # get the data and the fit parameters
        data, config = load_all_data(path, pressure=pressure)
        f.append([np.mean(data.alpha_cell), np.nanmedian(data.k_cell)])
        #f.append([np.mean(data.alpha_cell), np.mean(np.log10(data.k_cell))])

    fit_data.append(f)

fit_data = np.array(fit_data)

# plot the fit data in the two different plots
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.errorbar(pressures, np.mean(fit_data[:, :, i], axis=1),
                 np.std(fit_data[:, :, i], axis=1) / np.sqrt(fit_data[:, :, i].shape[1]), capsize=3,
                 label="NIH 3T3",
                 )


#plt.ylim(top=130)
plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(15.980000/2.54, 5.970000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(0.0, 3.2)
plt.figure(1).axes[0].set_ylim(0.0, 0.5)
plt.figure(1).axes[0].set_xticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[0].set_xticklabels(["0", "1", "2", "3"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].set_position([0.578804, 0.237068, 0.391927, 0.689799])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.217850, 0.995162])
plt.figure(1).axes[0].texts[0].set_text("b")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("alpha")
plt.figure(1).axes[1].set_xlim(0.0, 3.2)
plt.figure(1).axes[1].set_ylim(0.0, 195.6361353790945)
plt.figure(1).axes[1].set_xticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[1].set_xticklabels(["0", "1", "2", "3"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].set_position([0.103661, 0.215908, 0.391927, 0.710959])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].get_legend()._set_loc((1.789642, 0.629243))
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.233302, 0.995306])
plt.figure(1).axes[1].texts[0].set_text("a")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("stiffness (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()