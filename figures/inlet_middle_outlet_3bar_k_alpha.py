# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import fitStiffness, load_all_data, plotDensityScatter, plotBinnedData, plotStressStrain

import pylustrator
pylustrator.start()

position_list = []
for position in ["inlet", "middle", "outlet"]:
    pressure = 3
    data, config = load_all_data([
        fr"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\{position}\[0-9]\*_result.txt",
        fr"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\{position}\[0-9]\*_result.txt",
                ], pressure=pressure)
    position_list.append([data, config])

for i in range(3):
    data, config = position_list[i]

    plt.subplot(2, 3, 1+i)
    plt.title(["inlet", "middle", "outlet"][i])
    plotDensityScatter(np.abs(data.rp), np.log10(data.k_cell), skip=1, y_factor=0.33)
    plotBinnedData(np.abs(data.rp), np.log10(data.k_cell), np.arange(0, 90, 10), bin_func=np.median, error_func="quantiles")

    plt.subplot(2, 3, 3+1+i)

    plotDensityScatter(np.abs(data.rp), data.alpha_cell, skip=1, y_factor=0.33)
    plotBinnedData(np.abs(data.rp), data.alpha_cell, np.arange(0, 90, 10), bin_func=np.median, error_func="quantiles")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.250000/2.54, 8.000000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(-4.183295591939136, 88.04517013968203)
plt.figure(1).axes[0].set_ylim(1.0, 3.0)
plt.figure(1).axes[0].set_yticks([1.0, 2.0, 3.0])
plt.figure(1).axes[0].set_xticklabels([])
plt.figure(1).axes[0].set_yticklabels(["1.0", "2.0", "3.0"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[0].set_position([0.100418, 0.557717, 0.251213, 0.348164])
plt.figure(1).axes[0].set_xlabel('')
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].title.set_fontsize(10)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_position([-0.251907, 1.070730])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("stiffness k")
plt.figure(1).axes[1].set_xlim(-4.183295591939136, 88.04517013968203)
plt.figure(1).axes[1].set_ylim(0.0, 1.0)
plt.figure(1).axes[1].set_position([0.100418, 0.139922, 0.251213, 0.348164])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.251907, 1.001129])
plt.figure(1).axes[1].texts[0].set_text("d")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("position in channel (µm)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("fluidity $\\alpha$")
plt.figure(1).axes[2].set_xlim(-4.183295591939136, 88.04517013968203)
plt.figure(1).axes[2].set_ylim(1.0, 3.0)
plt.figure(1).axes[2].set_xticklabels([])
plt.figure(1).axes[2].set_yticklabels([])
plt.figure(1).axes[2].set_position([0.401873, 0.557717, 0.251213, 0.348164])
plt.figure(1).axes[2].set_xlabel('')
plt.figure(1).axes[2].set_ylabel('')
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].title.set_fontsize(9)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_position([-0.125536, 1.070730])
plt.figure(1).axes[2].texts[0].set_text("b")
plt.figure(1).axes[2].texts[0].set_weight("bold")
plt.figure(1).axes[2].title.set_fontsize(10)
plt.figure(1).axes[3].set_ylim(0.0, 1.0)
plt.figure(1).axes[3].set_yticklabels([])
plt.figure(1).axes[3].set_position([0.401873, 0.139922, 0.251213, 0.348164])
plt.figure(1).axes[3].set_ylabel('')
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_position([-0.125536, 1.001129])
plt.figure(1).axes[3].texts[0].set_text("e")
plt.figure(1).axes[3].texts[0].set_weight("bold")
plt.figure(1).axes[3].get_xaxis().get_label().set_text("position in channel (µm)")
plt.figure(1).axes[4].set_xlim(-4.183295591939136, 88.04517013968203)
plt.figure(1).axes[4].set_ylim(1.0, 3.0)
plt.figure(1).axes[4].set_xticklabels([])
plt.figure(1).axes[4].set_yticklabels([])
plt.figure(1).axes[4].set_position([0.703330, 0.557717, 0.251213, 0.348164])
plt.figure(1).axes[4].set_xlabel('')
plt.figure(1).axes[4].set_ylabel('')
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].title.set_fontsize(10)
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_position([-0.106581, 1.070730])
plt.figure(1).axes[4].texts[0].set_text("c")
plt.figure(1).axes[4].texts[0].set_weight("bold")
plt.figure(1).axes[5].set_ylim(0.0, 1.0)
plt.figure(1).axes[5].set_yticklabels([])
plt.figure(1).axes[5].set_position([0.703330, 0.139922, 0.251213, 0.348164])
plt.figure(1).axes[5].set_ylabel('')
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([-0.106581, 1.001129])
plt.figure(1).axes[5].texts[0].set_text("f")
plt.figure(1).axes[5].texts[0].set_weight("bold")
plt.figure(1).axes[5].get_xaxis().get_label().set_text("position in channel (µm)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()
