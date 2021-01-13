# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, get_cell_properties
from deformationcytometer.evaluation.helper_functions import plot_velocity_fit, plot_density_hist, \
    plotDensityLevels, plotBinnedData
import numpy as np

import pylustrator
pylustrator.start()

numbers = []
data_list = []
config_list = []
for index, pressure in enumerate([1, 2, 3]):
    data, config = load_all_data([
#    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
#    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
     r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\*\*_result.txt",
    ], pressure=pressure)
    data_list.append(data)
    config_list.append(config)

ranges = [
    np.arange(10, 80, 20),
    np.arange(10, 180, 20),
    np.arange(10, 280, 20),
]
colors = [
    "#d68800",
    "#d40000",
    "#b00038",
]

for index, pressure in enumerate([1, 2, 3]):
    ax = plt.subplot(2, 3, index+1)
    data = data_list[index]
    config = config_list[index]

    plt.loglog(data.omega, data.Gp1, "o", alpha=0.25, label="G'")
    plt.loglog(data.omega, data.Gp2, "o", alpha=0.25, label="G''")
    plt.ylabel("G' / G''")
    plt.xlabel("angular frequency")

    ax = plt.subplot(2, 6, 6+2*index+1)
    plot_density_hist(np.log10(data.k_cell), color="C0")
    plt.xlabel("log10(k)")
    plt.ylabel("relative density")

    ax = plt.subplot(2, 6, 6+2*index+2)
    plt.xlim(0, 1)
    plot_density_hist(data.alpha_cell, color="C1")
    plt.xlabel("alpha")

    numbers.append(len(data.rp))
    print(config)

print("numbers", numbers)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.250000/2.54, 9.000000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(3.0, 140.72848219623265)
plt.figure(1).axes[0].set_ylim(10.0, 5183.4642318175)
plt.figure(1).axes[0].legend()
plt.figure(1).axes[0].set_position([0.086243, 0.536754, 0.264021, 0.441165])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend()._set_loc((0.680252, 0.148377))
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.214366, 0.956803])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_ha("center")
plt.figure(1).axes[0].texts[1].set_position([0.500000, 0.913771])
plt.figure(1).axes[0].texts[1].set_text("1 bar")
plt.figure(1).axes[1].set_xlim(0.0, 4.0)
plt.figure(1).axes[1].set_ylim(0.0, 3.0)
plt.figure(1).axes[1].set_xticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[1].set_yticks([0.0, 2.0])
plt.figure(1).axes[1].set_xticklabels(["$10^0$", "", "$10^2$", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[1].set_yticklabels(["", ""], fontsize=10)
plt.figure(1).axes[1].set_position([0.086243, 0.152804, 0.120010, 0.214814])
plt.figure(1).axes[1].set_xticklabels([""], minor=True)
plt.figure(1).axes[1].set_xticks([np.nan], minor=True)
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.208234, 0.982666])
plt.figure(1).axes[1].texts[0].set_text("e")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("k")
plt.figure(1).axes[2].set_xlim(0.0, 1.0)
plt.figure(1).axes[2].set_xticks([0.0, 0.5, 1.0])
plt.figure(1).axes[2].set_xticklabels(["0", "0.5", "1"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[2].set_yticklabels([])
plt.figure(1).axes[2].set_position([0.230255, 0.152804, 0.120010, 0.214814])
plt.figure(1).axes[2].set_ylabel('')
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].xaxis.labelpad = 7.520000
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_position([-0.100000, 0.982666])
plt.figure(1).axes[2].texts[0].set_text("f")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("$\\alpha$")
plt.figure(1).axes[3].set_xlim(3.0, 140.72848219623265)
plt.figure(1).axes[3].set_ylim(10.0, 5183.4642318175)
plt.figure(1).axes[3].set_yticks([10.0, 100.0, 1000.0])
plt.figure(1).axes[3].set_yticklabels([])
plt.figure(1).axes[3].set_position([0.403069, 0.536754, 0.264021, 0.441165])
plt.figure(1).axes[3].set_ylabel('')
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_position([-0.153047, 0.956803])
plt.figure(1).axes[3].texts[0].set_text("b")
plt.figure(1).axes[3].texts[0].set_weight("bold")
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[1].new
plt.figure(1).axes[3].texts[1].set_ha("center")
plt.figure(1).axes[3].texts[1].set_position([0.500000, 0.913771])
plt.figure(1).axes[3].texts[1].set_text("2 bar")
plt.figure(1).axes[3].get_yaxis().get_label().set_text("")
plt.figure(1).axes[4].set_xlim(0.0, 4.0)
plt.figure(1).axes[4].set_ylim(0.0, 3.0)
plt.figure(1).axes[4].set_xticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[4].set_yticks([0.0, 2.0])
plt.figure(1).axes[4].set_xticklabels(["$10^0$", "", "$10^2$", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[4].set_yticklabels(["", ""], fontsize=10)
plt.figure(1).axes[4].set_position([0.403069, 0.152804, 0.120103, 0.214814])
plt.figure(1).axes[4].set_xticklabels([""], minor=True)
plt.figure(1).axes[4].set_xticks([np.nan], minor=True)
plt.figure(1).axes[4].set_ylabel('')
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_position([-0.153766, 0.982666])
plt.figure(1).axes[4].texts[0].set_text("g")
plt.figure(1).axes[4].texts[0].set_weight("bold")
plt.figure(1).axes[4].get_xaxis().get_label().set_text("k")
plt.figure(1).axes[5].set_xlim(0.0, 1.0)
plt.figure(1).axes[5].set_xticks([0.0, 0.5, 1.0])
plt.figure(1).axes[5].set_xticklabels(["0", "0.5", "1"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[5].set_yticklabels([])
plt.figure(1).axes[5].set_position([0.547192, 0.152804, 0.120103, 0.214814])
plt.figure(1).axes[5].set_ylabel('')
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].xaxis.labelpad = 6.080000
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([-0.180726, 0.982666])
plt.figure(1).axes[5].texts[0].set_text("h")
plt.figure(1).axes[5].texts[0].set_weight("bold")
plt.figure(1).axes[5].get_xaxis().get_label().set_text("$\\alpha$")
plt.figure(1).axes[6].set_xlim(3.0, 140.72848219623265)
plt.figure(1).axes[6].set_ylim(10.0, 5183.4642318175)
plt.figure(1).axes[6].set_yticklabels([])
plt.figure(1).axes[6].set_position([0.719895, 0.536754, 0.264021, 0.441165])
plt.figure(1).axes[6].set_ylabel('')
plt.figure(1).axes[6].spines['right'].set_visible(False)
plt.figure(1).axes[6].spines['top'].set_visible(False)
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[0].new
plt.figure(1).axes[6].texts[0].set_position([-0.131585, 0.956803])
plt.figure(1).axes[6].texts[0].set_text("c")
plt.figure(1).axes[6].texts[0].set_weight("bold")
plt.figure(1).axes[6].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[6].transAxes)  # id=plt.figure(1).axes[6].texts[1].new
plt.figure(1).axes[6].texts[1].set_ha("center")
plt.figure(1).axes[6].texts[1].set_position([0.500000, 0.913771])
plt.figure(1).axes[6].texts[1].set_text("3 bar")
plt.figure(1).axes[7].set_xlim(0.0, 4.0)
plt.figure(1).axes[7].set_ylim(0.0, 3.0)
plt.figure(1).axes[7].set_xticks([0.0, 1.0, 2.0, 3.0])
plt.figure(1).axes[7].set_yticks([0.0, 2.0])
plt.figure(1).axes[7].set_xticklabels(["$10^0$", "", "$10^2$", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[7].set_yticklabels(["", ""], fontsize=10)
plt.figure(1).axes[7].set_position([0.719640, 0.152804, 0.120717, 0.214814])
plt.figure(1).axes[7].set_xticklabels([""], minor=True)
plt.figure(1).axes[7].set_xticks([np.nan], minor=True)
plt.figure(1).axes[7].set_ylabel('')
plt.figure(1).axes[7].spines['right'].set_visible(False)
plt.figure(1).axes[7].spines['top'].set_visible(False)
plt.figure(1).axes[7].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[7].transAxes)  # id=plt.figure(1).axes[7].texts[0].new
plt.figure(1).axes[7].texts[0].set_position([-0.130322, 0.982666])
plt.figure(1).axes[7].texts[0].set_text("i")
plt.figure(1).axes[7].texts[0].set_weight("bold")
plt.figure(1).axes[7].get_xaxis().get_label().set_text("k")
plt.figure(1).axes[8].set_xlim(0.0, 1.0)
plt.figure(1).axes[8].set_xticks([0.0, 0.5, 1.0])
plt.figure(1).axes[8].set_xticklabels(["0", "0.5", "1"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[8].set_yticklabels([])
plt.figure(1).axes[8].set_position([0.864501, 0.152804, 0.120717, 0.214814])
plt.figure(1).axes[8].set_ylabel('')
plt.figure(1).axes[8].spines['right'].set_visible(False)
plt.figure(1).axes[8].spines['top'].set_visible(False)
plt.figure(1).axes[8].xaxis.labelpad = 5.360000
plt.figure(1).axes[8].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[8].transAxes)  # id=plt.figure(1).axes[8].texts[0].new
plt.figure(1).axes[8].texts[0].set_position([-0.104338, 0.982666])
plt.figure(1).axes[8].texts[0].set_text("j")
plt.figure(1).axes[8].texts[0].set_weight("bold")
plt.figure(1).axes[8].get_xaxis().get_label().set_text("$\\alpha$")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


