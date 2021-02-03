# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties#, load_all_data_new
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData, plot_joint_density
import numpy as np

import pylustrator
pylustrator.start()

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
    "Paclitaxel": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Paclitaxel\[2-9]\*_result.txt",
    "Y27632": r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\december_2020\2020_12_14_NIH3T3_drugs\Y27632\[2-9]\*_result.txt",
}

drugs = drugs_dez

ax_k = []
ax_a = []
# iterate over all times
for index, name in enumerate(drugs.keys()):
    print(index, name)
    # get the data and the fit parameters
    data, config = load_all_data(drugs[name], pressure=3)

    ax_k.append(plt.subplot(4, 6, index*2+1))
    #plot_joint_density(data.k_cell, data.alpha_cell, label=name, only_kde=True)
    plt.title(name)
    plot_density_hist(np.log10(data.k_cell))
    plt.xlim(0, 5)
    plt.xticks(np.arange(5))
    plt.grid()

    ax_a.append(plt.subplot(4, 6, index*2+1+1))
    #plot_joint_density(data.k_cell, data.alpha_cell, label=name, only_kde=True)
    plot_density_hist(data.alpha_cell, color="C1")
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1, 0.2))
    plt.grid()

pylustrator.helper_functions.axes_to_grid(ax_k)
pylustrator.helper_functions.axes_to_grid(ax_a)

if 0:
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).axes[0].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[0].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[0].set_xticklabels([])
    plt.figure(1).axes[0].grid(True)
    plt.figure(1).axes[0].set_position([0.125000, 0.653529, 0.110790, 0.177362])
    plt.figure(1).axes[0].set_xlabel('')
    plt.figure(1).axes[0].spines['right'].set_visible(False)
    plt.figure(1).axes[0].spines['top'].set_visible(False)
    plt.figure(1).axes[1].set_xlim(0.0, 1.0)
    plt.figure(1).axes[1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[1].set_xticklabels([])
    plt.figure(1).axes[1].grid(True)
    plt.figure(1).axes[1].set_position([0.257857, 0.653529, 0.110790, 0.177362])
    plt.figure(1).axes[1].set_xlabel('')
    plt.figure(1).axes[1].spines['right'].set_visible(False)
    plt.figure(1).axes[1].spines['top'].set_visible(False)
    plt.figure(1).axes[2].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[2].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[2].set_xticklabels([])
    plt.figure(1).axes[2].set_yticklabels([])
    plt.figure(1).axes[2].grid(True)
    plt.figure(1).axes[2].set_position([0.390714, 0.653529, 0.110790, 0.177362])
    plt.figure(1).axes[2].set_xlabel('')
    plt.figure(1).axes[2].set_ylabel('')
    plt.figure(1).axes[2].spines['right'].set_visible(False)
    plt.figure(1).axes[2].spines['top'].set_visible(False)
    plt.figure(1).axes[3].set_xlim(0.0, 1.0)
    plt.figure(1).axes[3].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[3].set_xticklabels([])
    plt.figure(1).axes[3].set_yticklabels([])
    plt.figure(1).axes[3].grid(True)
    plt.figure(1).axes[3].set_position([0.523571, 0.653529, 0.110790, 0.177362])
    plt.figure(1).axes[3].set_xlabel('')
    plt.figure(1).axes[3].set_ylabel('')
    plt.figure(1).axes[3].spines['right'].set_visible(False)
    plt.figure(1).axes[3].spines['top'].set_visible(False)
    plt.figure(1).axes[4].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[4].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[4].set_xticklabels([])
    plt.figure(1).axes[4].set_yticklabels([])
    plt.figure(1).axes[4].grid(True)
    plt.figure(1).axes[4].set_position([0.656429, 0.653529, 0.110790, 0.177362])
    plt.figure(1).axes[4].set_xlabel('')
    plt.figure(1).axes[4].set_ylabel('')
    plt.figure(1).axes[4].spines['right'].set_visible(False)
    plt.figure(1).axes[4].spines['top'].set_visible(False)
    plt.figure(1).axes[5].set_xlim(0.0, 1.0)
    plt.figure(1).axes[5].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[5].set_xticklabels([])
    plt.figure(1).axes[5].set_yticklabels([])
    plt.figure(1).axes[5].grid(True)
    plt.figure(1).axes[5].set_position([0.789286, 0.653529, 0.110790, 0.177362])
    plt.figure(1).axes[5].set_xlabel('')
    plt.figure(1).axes[5].set_ylabel('')
    plt.figure(1).axes[5].spines['right'].set_visible(False)
    plt.figure(1).axes[5].spines['top'].set_visible(False)
    plt.figure(1).axes[6].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[6].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[6].set_xticklabels([])
    plt.figure(1).axes[6].grid(True)
    plt.figure(1).axes[6].set_position([0.125000, 0.381765, 0.110790, 0.177362])
    plt.figure(1).axes[6].set_xlabel('')
    plt.figure(1).axes[6].spines['right'].set_visible(False)
    plt.figure(1).axes[6].spines['top'].set_visible(False)
    plt.figure(1).axes[7].set_xlim(0.0, 1.0)
    plt.figure(1).axes[7].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[7].set_xticklabels([])
    plt.figure(1).axes[7].grid(True)
    plt.figure(1).axes[7].set_position([0.257857, 0.381765, 0.110790, 0.177362])
    plt.figure(1).axes[7].set_xlabel('')
    plt.figure(1).axes[7].spines['right'].set_visible(False)
    plt.figure(1).axes[7].spines['top'].set_visible(False)
    plt.figure(1).axes[8].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[8].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[8].set_xticklabels([])
    plt.figure(1).axes[8].set_yticklabels([])
    plt.figure(1).axes[8].grid(True)
    plt.figure(1).axes[8].set_position([0.390714, 0.381765, 0.110790, 0.177362])
    plt.figure(1).axes[8].set_xlabel('')
    plt.figure(1).axes[8].set_ylabel('')
    plt.figure(1).axes[8].spines['right'].set_visible(False)
    plt.figure(1).axes[8].spines['top'].set_visible(False)
    plt.figure(1).axes[9].set_xlim(0.0, 1.0)
    plt.figure(1).axes[9].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[9].set_xticklabels([])
    plt.figure(1).axes[9].set_yticklabels([])
    plt.figure(1).axes[9].grid(True)
    plt.figure(1).axes[9].set_position([0.523571, 0.381765, 0.110790, 0.177362])
    plt.figure(1).axes[9].set_xlabel('')
    plt.figure(1).axes[9].set_ylabel('')
    plt.figure(1).axes[9].spines['right'].set_visible(False)
    plt.figure(1).axes[9].spines['top'].set_visible(False)
    plt.figure(1).axes[10].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[10].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[10].set_xticklabels([])
    plt.figure(1).axes[10].set_yticklabels([])
    plt.figure(1).axes[10].grid(True)
    plt.figure(1).axes[10].set_position([0.656429, 0.381765, 0.110790, 0.177362])
    plt.figure(1).axes[10].set_xlabel('')
    plt.figure(1).axes[10].set_ylabel('')
    plt.figure(1).axes[10].spines['right'].set_visible(False)
    plt.figure(1).axes[10].spines['top'].set_visible(False)
    plt.figure(1).axes[11].set_xlim(0.0, 1.0)
    plt.figure(1).axes[11].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[11].set_xticklabels([])
    plt.figure(1).axes[11].set_yticklabels([])
    plt.figure(1).axes[11].grid(True)
    plt.figure(1).axes[11].set_position([0.789286, 0.381765, 0.110790, 0.177362])
    plt.figure(1).axes[11].set_xlabel('')
    plt.figure(1).axes[11].set_ylabel('')
    plt.figure(1).axes[11].spines['right'].set_visible(False)
    plt.figure(1).axes[11].spines['top'].set_visible(False)
    plt.figure(1).axes[12].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[12].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[12].set_xticklabels(["0", "1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[12].grid(True)
    plt.figure(1).axes[12].set_position([0.125000, 0.110000, 0.110790, 0.177362])
    plt.figure(1).axes[12].spines['right'].set_visible(False)
    plt.figure(1).axes[12].spines['top'].set_visible(False)
    plt.figure(1).axes[13].set_xlim(0.0, 1.0)
    plt.figure(1).axes[13].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[13].set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[13].grid(True)
    plt.figure(1).axes[13].set_position([0.257857, 0.110000, 0.110790, 0.177362])
    plt.figure(1).axes[13].spines['right'].set_visible(False)
    plt.figure(1).axes[13].spines['top'].set_visible(False)
    plt.figure(1).axes[14].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[14].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[14].set_xticklabels(["0", "1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[14].set_yticklabels([])
    plt.figure(1).axes[14].grid(True)
    plt.figure(1).axes[14].set_position([0.390714, 0.110000, 0.110790, 0.177362])
    plt.figure(1).axes[14].set_ylabel('')
    plt.figure(1).axes[14].spines['right'].set_visible(False)
    plt.figure(1).axes[14].spines['top'].set_visible(False)
    plt.figure(1).axes[15].set_xlim(0.0, 1.0)
    plt.figure(1).axes[15].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[15].set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[15].set_yticklabels([])
    plt.figure(1).axes[15].grid(True)
    plt.figure(1).axes[15].set_position([0.523571, 0.110000, 0.110790, 0.177362])
    plt.figure(1).axes[15].set_ylabel('')
    plt.figure(1).axes[15].spines['right'].set_visible(False)
    plt.figure(1).axes[15].spines['top'].set_visible(False)
    plt.figure(1).axes[16].set_xlim(0.0, 5.085116043516992)
    plt.figure(1).axes[16].set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    plt.figure(1).axes[16].set_xticklabels(["0", "1", "2", "3", "4", "5"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[16].set_yticklabels([])
    plt.figure(1).axes[16].grid(True)
    plt.figure(1).axes[16].set_position([0.656429, 0.110000, 0.110790, 0.177362])
    plt.figure(1).axes[16].set_ylabel('')
    plt.figure(1).axes[16].spines['right'].set_visible(False)
    plt.figure(1).axes[16].spines['top'].set_visible(False)
    plt.figure(1).axes[17].set_xlim(0.0, 1.0)
    plt.figure(1).axes[17].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, np.nan])
    plt.figure(1).axes[17].set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
    plt.figure(1).axes[17].set_yticklabels([])
    plt.figure(1).axes[17].grid(True)
    plt.figure(1).axes[17].set_position([0.789286, 0.110000, 0.110790, 0.177362])
    plt.figure(1).axes[17].set_ylabel('')
    plt.figure(1).axes[17].spines['right'].set_visible(False)
    plt.figure(1).axes[17].spines['top'].set_visible(False)
    #% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


