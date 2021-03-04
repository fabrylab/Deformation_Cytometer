# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties#, load_all_data_new
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData, plot_joint_density, split_axes, get_mode_stats, bootstrap_match_hist, load_all_data_new
import numpy as np

import pylustrator
pylustrator.start()
import glob
import natsort


files = natsort.natsorted(glob.glob(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_02_24_NIH3T3_LatrunculinB_doseresponse\*"))
#files = natsort.natsorted(glob.glob(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\january_2021\2021_03_01_NIH3T3_LatrunculinB_doseresponse\*"))
last_file = None
file_pairs = []
for file in files:
    if last_file is None:
        last_file = file
    else:
        print(file)
        #if file[-len("1000 nM - Kontrolle"):] != "1000 nM - Kontrolle":
        file_pairs.append([last_file+"\*_result.txt", file+"\*_result.txt"])
        last_file = None
print(file_pairs)

for row_index, pressure in enumerate([0.5, 1, 2, 3]):
    plt.subplot(4, 1, row_index+1)
    for index, paths in enumerate(file_pairs):
        try:
            data0, config0 = load_all_data_new(paths[0], solidity_threshold=0.7, irregularity_threshold=1.3)#, pressure=pressure)
        except (FileNotFoundError, ValueError) as err:
            print("ERROR", paths[0], err)
            continue
        try:
            data1, config1 = load_all_data_new(paths[1], solidity_threshold=0.7, irregularity_threshold=1.3)#, pressure=pressure)
        except (FileNotFoundError, ValueError) as err:
            print("ERROR", paths[1], err)
            continue

        ratios = []
        # bootstrap with matched histograms
        for i in range(10):
            data0b, data1b = bootstrap_match_hist([data0, data1], bin_width=25, max_bin=300, property="stress")
            ratios.append(get_mode_stats(data0b.k_cell)[0]/get_mode_stats(data1b.k_cell)[0])

        # print the cell counts
        print(np.sum(~np.isnan(data0.k_cell)))
        print(np.sum(~np.isnan(data1.k_cell)))
        #plt.subplot(4, len(file_pairs), row_index*len(file_pairs) + 1 + index)

        try:
            #plot_density_hist(np.log10(data0.k_cell))
            #plot_density_hist(np.log10(data1.k_cell))
            #plt.errorbar(index, np.mean(stat_k0[:, 0]), np.std(stat_k0[:, 0]))

            plt.errorbar(index, np.mean(ratios), np.std(ratios), fmt="o")

        except np.linalg.LinAlgError:
            pass
    break

plt.legend()

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_xlim(-0.25, 6.5)
plt.figure(1).axes[0].set_ylim(0.0, 1.2818077551966356)
plt.figure(1).axes[0].set_xticks([0.0, 2.0, 4.0, 6.0])
plt.figure(1).axes[0].set_xticklabels(["1", "10", "100", "1000"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].grid(True)
plt.figure(1).axes[0].set_position([0.129688, 0.619834, 0.314342, 0.358083])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_xaxis().get_label().set_text('lat B concentration')
plt.figure(1).axes[0].get_yaxis().get_label().set_text("0.5 bar\nrelative stiffness")
plt.figure(1).axes[1].set_xlim(-0.25, 6.5)
plt.figure(1).axes[1].set_ylim(0.0, 1.2818077551966356)
plt.figure(1).axes[1].set_xticks([0.0, 2.0, 4.0, 6.0])
plt.figure(1).axes[1].set_xticklabels(["1", "10", "100", "1000"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].grid(True)
plt.figure(1).axes[1].set_position([0.621596, 0.619834, 0.314342, 0.358083])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].get_xaxis().get_label().set_text('lat B concentration')
plt.figure(1).axes[1].get_yaxis().get_label().set_text("1 bar\nrelative stiffness")
plt.figure(1).axes[2].set_xlim(-0.25, 6.5)
plt.figure(1).axes[2].set_ylim(0.0, 1.2818077551966356)
plt.figure(1).axes[2].set_xticks([0.0, 2.0, 4.0, 6.0])
plt.figure(1).axes[2].set_xticklabels(["1", "10", "100", "1000"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[2].grid(True)
plt.figure(1).axes[2].set_position([0.129688, 0.114583, 0.314342, 0.358083])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].get_xaxis().get_label().set_text('lat B concentration')
plt.figure(1).axes[2].get_yaxis().get_label().set_text("2 bar\nrelative stiffness")
plt.figure(1).axes[3].set_xlim(-0.25, 6.5)
plt.figure(1).axes[3].set_ylim(0.0, 1.2818077551966356)
plt.figure(1).axes[3].set_xticks([0.0, 2.0, 4.0, 6.0])
plt.figure(1).axes[3].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
plt.figure(1).axes[3].set_xticklabels(["1", "10", "100", "1000"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[3].set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00", "1.25"], fontsize=10)
plt.figure(1).axes[3].grid(True)
plt.figure(1).axes[3].set_position([0.621596, 0.114583, 0.314342, 0.358083])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].get_xaxis().get_label().set_text("lat B concentration")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("3 bar\nrelative stiffness")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


