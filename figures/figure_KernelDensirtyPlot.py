# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties#, load_all_data_new
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData, plot_joint_density
import numpy as np

import pylustrator
pylustrator.start()

THP1 = [
    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\Mar THP1\Control1\T20\*_result.txt",
]

K562 = [
    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_K562_0%FCS_time\2\*_result.txt",
    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\2\*_result.txt",
]

NIH3T3 = [
    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\2\*_result.txt",
    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\2\*_result.txt",
    rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\2\*_result.txt",
]


alg25 = [
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_24_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_27_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_14_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_28_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_30_alginate2.5%_dmem_NIH_3T3\*\*_result.txt",
]

alg20 = [
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_27_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_28_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_28_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_30_alginate2.0%_dmem_NIH_3T3\*\*_result.txt",
]

alg15 = [
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_07_30_alginate1.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_14_alginate1.5%_dmem_NIH_3T3\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_14_alginate1.5%_dmem_NIH_3T3_2\*\*_result.txt",
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\evaluation\diff % alginate\2020_10_28_alginate1.5%_dmem_NIH_3T3\*\*_result.txt",
]

if 0:
    plt.subplot(2, 3, 1, label="cells")
    # iterate over all times
    for index, path in enumerate([THP1, K562, NIH3T3]):
        # get the data and the fit parameters
        data, config = load_all_data(path, pressure=2)

        plot_joint_density(np.log10(data.k_cell), data.alpha_cell, label=["THP1", "K562", "NIH3T3"][index])
        plt.xlim(1, 3)
        plt.ylim(0, 1)

    plt.legend(fontsize=8)

    plt.subplot(2, 3, 2, label="pressure")
    # iterate over all times
    for pressure in [1, 2, 3]:
        # get the data and the fit parameters
        data, config = load_all_data(NIH3T3, pressure=pressure)

        plot_joint_density(np.log10(data.k_cell), data.alpha_cell, color=pylustrator.lab_colormap.LabColormap(["C2", "C3"], 3)(pressure-1), label=f"{pressure} bar")
        plt.xlim(1, 3)
        plt.ylim(0, 1)

    plt.legend(fontsize=8)

    plt.subplot(2, 3, 3, label="time")
    # iterate over all times
    for index, time in enumerate(range(2, 13, 2)):
        # get the data and the fit parameters
        data, config = load_all_data([
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
    #        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1\{time}\*_result.txt",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_2\{time}\*_result.txt",
        ], repetition=1)

        plot_joint_density(np.log10(data.k_cell), data.alpha_cell, label=f"{time*5} min", color=pylustrator.lab_colormap.LabColormap(["C2", "C3"], len(range(2, 13, 2)))(index), only_kde=True)
        plt.xlim(1, 3)
        plt.ylim(0, 1)

    plt.legend(fontsize=8)

if 0:
    plt.subplot(2, 3, 4, label="alginat")
    # iterate over all times
    for index in range(3):
        # get the data and the fit parameters
        data, config = load_all_data([alg15, alg20, alg25][index], pressure=index+1)

        plot_joint_density(np.log10(data.k_cell), data.alpha_cell, label=["1.5%", "2.0%", "2.5%"][index], color=pylustrator.lab_colormap.LabColormap(["C2", "C3"], 3)(index), only_kde=True)
        plt.xlim(1, 3)
        plt.ylim(0, 1)

    plt.legend(fontsize=8)

plt.subplot(2, 3, 5, label="drugs")
drugs = {
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
# iterate over all times
for index, name in enumerate(drugs.keys()):
    # get the data and the fit parameters
    data, config = load_all_data(drugs[name], pressure=3)

    plot_joint_density(np.log10(data.k_cell), data.alpha_cell, label=name, only_kde=True)
    #plot_joint_density(data.mu1, data.eta1, label=name, only_kde=True)
    plt.xlim(1, 3)
    plt.ylim(0, 1)

plt.legend(fontsize=8)


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).ax_dict["drugs"].legend(fontsize=10, title_fontsize=10.0)
plt.figure(1).ax_dict["drugs"].set_position([0.137592, 0.110000, 0.477353, 0.610000])
plt.figure(1).ax_dict["drugs"].set_zorder(1)
plt.figure(1).ax_dict["drugs"].get_legend()._set_loc((1.285690, 0.574192))
plt.figure(1).ax_dict["drugs"].get_legend()._set_loc((1.141667, 0.574192))
plt.figure(1).ax_dict["drugs"].get_xaxis().get_label().set_text("log10(k)")
plt.figure(1).ax_dict["drugs"].get_yaxis().get_label().set_text("alpha")
plt.figure(1).ax_dict["drugs_right"].set_position([0.614945, 0.110000, 0.119338, 0.610000])
plt.figure(1).ax_dict["drugs_top"].set_position([0.137592, 0.720000, 0.477353, 0.152500])
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


