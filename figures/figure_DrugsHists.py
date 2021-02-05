# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties#, load_all_data_new
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData, plot_joint_density
import numpy as np

import pylustrator
pylustrator.start()

def get_mode_stats(x):
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    from scipy import stats

    x = np.array(x)
    x = x[~np.isnan(x)]

    def get_mode(x):
        kde = stats.gaussian_kde(x)
        return x[np.argmax(kde(x))]

    mode = get_mode(x)
    err = bootstrap_error(x, get_mode, repetitions=10)
    def string(x):
        if x > 1:
            return str(round(x))
        else:
            return str(round(x, 2))
    plt.text(0.5, 1, string(mode)+"$\pm$"+string(err), transform=plt.gca().transAxes, ha="center", va="top")
    return mode, err, len(x)

drugs_jan = {
    "blebistatin": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\Blebbistatin\[2-9]\*_result.txt",
    "Cytochalasin D": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\Cytochalasin D\[2-9]\*_result.txt",
    "Dibuturyl cAMP": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\Dibuturyl cAMP\[2-9]\*_result.txt",
    "DMSO 1%": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\DMSO 1%\[2-9]\*_result.txt",
    "Kontrolle": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\Kontrolle\[2-9]\*_result.txt",
    "Latrunculin B": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\Latrunculin B\[2-9]\*_result.txt",
    "Nocodazol": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\Nocodazol\[2-9]\*_result.txt",
    "Paclitaxel": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\Paclitaxel\[2-9]\*_result.txt",
    "Y27632": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_01_27_NIH3T3_drugs\Y27632\[2-9]\*_result.txt",
}

drugs_dez = {
    "Blebbistatin": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Blebbistatin\[2-9]\*_result.txt",
    "Cytochalasin D": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Cytochalasin D\[2-9]\*_result.txt",
    "Dibuturyl cAMP": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Dibuturyl cAMP\[2-9]\*_result.txt",
    "DMSO 1%": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\DMSO 1%\[2-9]\*_result.txt",
    "DMSO 2%": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\DMSO 2%\[2-9]\*_result.txt",
    "Kontrolle": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Kontrolle\[2-9]\*_result.txt",
    "Latrunculin A": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Latrunculin A\[2-9]\*_result.txt",
    "Nocodazole": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Nocodazole\[2-9]\*_result.txt",
    "Paclitaxel": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Paclitaxel\[2-9]\*_result.txt",    "Y27632": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Y27632\[2-9]\*_result.txt",
}

drugs = drugs_jan

drugs = {}
import glob
from pathlib import Path
#for folder in glob.glob(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\desmin_cells\2020_12_08_desmin_cytoD\*\*\\"):
#    folder = Path(folder)
#    name = str(folder.parent.name)+"_"+str(folder.name)
#    drugs[name] = str(folder)+"\*_result.txt"

import natsort
for folder in natsort.natsorted(glob.glob(rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_02_03_NIH3T3_LatB_drugresponse\*\\")):
    folder = Path(folder)
    name = str(folder.name)
    drugs[name] = str(folder)+"\*\*_result.txt"

N = len(drugs)
cols = int(np.sqrt(N))
rows = int(np.ceil(N/cols))

ax_k = []
ax_a = []
# iterate over all times
for index, name in enumerate(drugs.keys()):
    plt.figure(1)
    # get the data and the fit parameters
    data, config = load_all_data(drugs[name])

    print(index, name, len(data), np.sum(~np.isnan(data.k_cell)))

    ax_k.append(plt.subplot(rows, cols*2, index*2+1))
    plt.title(name)
    plot_density_hist(np.log10(data.k_cell))
    stat_k = get_mode_stats(data.k_cell)
    plt.xlim(1, 4)
    plt.xticks(np.arange(5))
    plt.grid()

    ax_a.append(plt.subplot(rows, cols*2, index*2+1+1))

    plot_density_hist(data.alpha_cell, color="C1")
    stat_alpha = get_mode_stats(data.alpha_cell)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1, 0.2))

    plt.grid()

    plt.figure(2)
    plt.title("Januar Measured TT")
    plt.errorbar(stat_k[0], stat_alpha[0], xerr=stat_k[1], yerr=stat_alpha[1], label=name, color=plt.get_cmap("viridis")(index/10))#, color=f"C{int(index//3)}", alpha=[0.3, .6, 1][index%3])

plt.figure(2)
plt.legend()
plt.xlabel("k")
plt.ylabel("alpha")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.figure(1)
pylustrator.helper_functions.axes_to_grid(ax_k)
pylustrator.helper_functions.axes_to_grid(ax_a)

plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")


plt.show()


