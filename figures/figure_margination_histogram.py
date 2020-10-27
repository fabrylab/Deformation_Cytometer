# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data
import numpy as np
from scipy import stats

import pylustrator
pylustrator.start()

if 1:
    numbers = []
    for index, pressure in enumerate([1, 2, 3]):
        ax = plt.subplot(3, 3, index+1)

        #data, config = load_all_data(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data" +
        #                             r"\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\[0-9]\*_result.txt", pressure=pressure)
        data, config = load_all_data([
            r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
            r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
            ], pressure=pressure)

        plt.hist(data.rp, bins=np.linspace(-100, 100, 20), density=True, alpha=0.8)
        numbers.append(len(data.rp))

        kde = stats.gaussian_kde(np.hstack((data.rp, -data.rp)))
        xx = np.linspace(-110, 110, 1000)
        plt.plot(xx, kde(xx), "--k", lw=0.8)

    numbers = []
    for index, pressure in enumerate([1, 2, 3]):
        ax = plt.subplot(3, 3, 3+ index + 1)

        # data, config = load_all_data(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data" +
        #                             r"\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\[0-9]\*_result.txt", pressure=pressure)
        data, config = load_all_data([
            r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\middle\[0-9]\*_result.txt",
            r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\middle\[0-9]\*_result.txt",
        ], pressure=pressure)

        plt.hist(data.rp, bins=np.linspace(-100, 100, 20), density=True, alpha=0.8)
        numbers.append(len(data.rp))

        kde = stats.gaussian_kde(np.hstack((data.rp, -data.rp)))
        xx = np.linspace(-110, 110, 1000)
        plt.plot(xx, kde(xx), "--k", lw=0.8)

    numbers = []
    for index, pressure in enumerate([1, 2, 3]):
        ax = plt.subplot(3, 3, 6 + index + 1)

        # data, config = load_all_data(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data" +
        #                             r"\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\[0-9]\*_result.txt", pressure=pressure)
        data, config = load_all_data([
            r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\outlet\[0-9]\*_result.txt",
            r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\outlet\[0-9]\*_result.txt",
        ], pressure=pressure)

        plt.hist(data.rp, bins=np.linspace(-100, 100, 20), density=True, alpha=0.8)
        numbers.append(len(data.rp))

        kde = stats.gaussian_kde(np.hstack((data.rp, -data.rp)))
        xx = np.linspace(-110, 110, 1000)
        plt.plot(xx, kde(xx), "--k", lw=0.8)


    print("numbers", numbers)

#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(15.360000/2.54, 9.570000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.131821, 0.715252, 0.246468, 0.249562])
plt.figure(1).axes[0].set_xlim(-90.0, 90.0)
plt.figure(1).axes[0].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[0].set_xticks([np.nan], minor=True)
plt.figure(1).axes[0].set_ylim(0.0, 0.023)
plt.figure(1).axes[0].set_yticklabels(["0.00", "0.01", "0.02"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[0].set_yticks([0.0, 0.01, 0.02])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.509148, 0.891045])
plt.figure(1).axes[0].texts[0].set_text("1 bar")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("probability\ndensity")
plt.figure(1).axes[1].set_position([0.418491, 0.715252, 0.246468, 0.249562])
plt.figure(1).axes[1].set_xlim(-90.0, 90.0)
plt.figure(1).axes[1].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[1].set_xticks([np.nan], minor=True)
plt.figure(1).axes[1].set_ylim(0.0, 0.023)
plt.figure(1).axes[1].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[1].set_yticks([0.0, 0.01, 0.02])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].xaxis.labelpad = 3.716691
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.545742, 0.891045])
plt.figure(1).axes[1].texts[0].set_text("2 bar")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[2].set_position([0.706958, 0.715252, 0.246468, 0.249562])
plt.figure(1).axes[2].set_xlim(-90.0, 90.0)
plt.figure(1).axes[2].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[2].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[2].set_xticks([np.nan], minor=True)
plt.figure(1).axes[2].set_ylim(0.0, 0.023)
plt.figure(1).axes[2].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[2].set_yticks([0.0, 0.01, 0.02])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.504574, 0.891045])
plt.figure(1).axes[2].texts[0].set_text("3 bar")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[2].new
plt.figure(1).axes[2].texts[1].set_ha("center")
plt.figure(1).axes[2].texts[1].set_position([1.015452, 0.429650])
plt.figure(1).axes[2].texts[1].set_rotation(90.0)
plt.figure(1).axes[2].texts[1].set_text("inlet")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[3].set_position([0.131821, 0.433146, 0.246468, 0.249562])
plt.figure(1).axes[3].set_xlim(-90.0, 90.0)
plt.figure(1).axes[3].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[3].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[3].set_ylim(0.0, 0.023)
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].get_yaxis().get_label().set_text("probability\ndensity")
plt.figure(1).axes[4].set_position([0.418491, 0.433146, 0.246468, 0.249562])
plt.figure(1).axes[4].set_xlim(-90.0, 90.0)
plt.figure(1).axes[4].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[4].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[4].set_ylim(0.0, 0.023)
plt.figure(1).axes[4].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[4].set_yticks([0.0, 0.01, 0.02])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[5].set_position([0.706958, 0.433146, 0.246468, 0.249562])
plt.figure(1).axes[5].set_xlim(-90.0, 90.0)
plt.figure(1).axes[5].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[5].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[5].set_ylim(0.0, 0.023)
plt.figure(1).axes[5].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[5].set_yticks([0.0, 0.01, 0.02])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_ha("center")
plt.figure(1).axes[5].texts[0].set_position([1.015452, 0.375092])
plt.figure(1).axes[5].texts[0].set_rotation(90.0)
plt.figure(1).axes[5].texts[0].set_text("middle")
plt.figure(1).axes[6].set_position([0.131821, 0.151040, 0.246468, 0.249562])
plt.figure(1).axes[6].set_xlim(-90.0, 90.0)
plt.figure(1).axes[6].set_xticklabels(["-75", "0", "75"], fontsize=10)
plt.figure(1).axes[6].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[6].set_ylim(0.0, 0.023)
plt.figure(1).axes[6].spines['right'].set_visible(False)
plt.figure(1).axes[6].spines['top'].set_visible(False)
plt.figure(1).axes[6].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[6].get_yaxis().get_label().set_text("probability\ndensity")
plt.figure(1).axes[7].set_position([0.418491, 0.151040, 0.246468, 0.249562])
plt.figure(1).axes[7].set_xlim(-90.0, 90.0)
plt.figure(1).axes[7].set_xticklabels(["-75", "0", "75"], fontsize=10)
plt.figure(1).axes[7].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[7].set_ylim(0.0, 0.023)
plt.figure(1).axes[7].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[7].set_yticks([0.0, 0.01, 0.02])
plt.figure(1).axes[7].spines['right'].set_visible(False)
plt.figure(1).axes[7].spines['top'].set_visible(False)
plt.figure(1).axes[7].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[8].set_position([0.706958, 0.151040, 0.246468, 0.249562])
plt.figure(1).axes[8].set_xlim(-90.0, 90.0)
plt.figure(1).axes[8].set_xticklabels(["-75", "0", "75"], fontsize=10)
plt.figure(1).axes[8].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[8].set_ylim(0.0, 0.023)
plt.figure(1).axes[8].set_yticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[8].set_yticks([0.0, 0.01, 0.02])
plt.figure(1).axes[8].spines['right'].set_visible(False)
plt.figure(1).axes[8].spines['top'].set_visible(False)
plt.figure(1).axes[8].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[8].transAxes)  # id=plt.figure(1).axes[8].texts[0].new
plt.figure(1).axes[8].texts[0].set_ha("center")
plt.figure(1).axes[8].texts[0].set_position([1.015452, 0.399910])
plt.figure(1).axes[8].texts[0].set_rotation(90.0)
plt.figure(1).axes[8].texts[0].set_text("outlet")
plt.figure(1).axes[8].get_xaxis().get_label().set_text("radial position (µm)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")

plt.show()


