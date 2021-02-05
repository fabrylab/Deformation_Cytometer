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

# iterate over all times
for index, name in enumerate(drugs.keys()):
    print(index, name)
    plt.subplot(rows, cols, index+1)

    # get the data and the fit parameters
    data, config = load_all_data(drugs[name])

    #plot_joint_density(data.k_cell, data.alpha_cell, label=name, only_kde=True)
    plt.title(name)
    plotDensityScatter(data.stress, data.strain)
    plt.xlim(0, 300)
    plt.ylim(0, 1)

pylustrator.helper_functions.axes_to_grid()

if 0:
    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).axes[0].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[0].set_ylim(0.0, 1.2)
    plt.figure(1).axes[0].set_position([0.125000, 0.698969, 0.242378, 0.245404])
    plt.figure(1).axes[0].get_yaxis().get_label().set_text("strain")
    plt.figure(1).axes[1].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[1].set_ylim(0.0, 1.2)
    plt.figure(1).axes[1].set_position([0.415853, 0.698969, 0.242378, 0.245404])
    plt.figure(1).axes[1].get_xaxis().get_label().set_text("")
    plt.figure(1).axes[1].get_yaxis().get_label().set_text("")
    plt.figure(1).axes[2].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[2].set_ylim(0.0, 1.2)
    plt.figure(1).axes[2].set_position([0.706707, 0.698969, 0.242378, 0.245404])
    plt.figure(1).axes[3].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[3].set_ylim(0.0, 1.2)
    plt.figure(1).axes[3].set_position([0.125000, 0.404485, 0.242378, 0.245404])
    plt.figure(1).axes[3].get_yaxis().get_label().set_text("strain")
    plt.figure(1).axes[4].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[4].set_ylim(0.0, 1.2)
    plt.figure(1).axes[4].set_position([0.415853, 0.404485, 0.242378, 0.245404])
    plt.figure(1).axes[5].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[5].set_ylim(0.0, 1.2)
    plt.figure(1).axes[5].set_position([0.706707, 0.404485, 0.242378, 0.245404])
    plt.figure(1).axes[6].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[6].set_ylim(0.0, 1.2)
    plt.figure(1).axes[6].set_position([0.125000, 0.110000, 0.242378, 0.245404])
    plt.figure(1).axes[6].get_xaxis().get_label().set_text("stress")
    plt.figure(1).axes[6].get_yaxis().get_label().set_text("strain")
    plt.figure(1).axes[7].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[7].set_ylim(0.0, 1.2)
    plt.figure(1).axes[7].set_position([0.415853, 0.110000, 0.242378, 0.245404])
    plt.figure(1).axes[7].get_xaxis().get_label().set_text("stress")
    plt.figure(1).axes[8].set_xlim(0.0, 304.13529498542937)
    plt.figure(1).axes[8].set_ylim(0.0, 1.2)
    plt.figure(1).axes[8].set_position([0.706707, 0.110000, 0.242378, 0.245404])
    plt.figure(1).axes[8].get_xaxis().get_label().set_text("stress")
    #% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


