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


def get_k(data, config):
    def omega(x):
        return 0.25 * x / (2 * np.pi)

    p = fitStiffness(data, config)
    v = data.velocity_fitted
    w = omega(data.velocity_gradient)
    k = data.stress / ( (data.strain - p[2]) * (np.abs(w) + (v / (np.pi * 2 * config["imaging_pos_mm"]))) ** p[1])
    return k


for i in range(3):
    data, config = position_list[i]

    plt.subplot(1, 3, 1+i)
    k = get_k(data, config)

    #plotStressStrain(data, config)

    plotDensityScatter(np.abs(data.rp), k, skip=1, y_factor=0.33)
    plotBinnedData(np.abs(data.rp), k, np.arange(0, 90, 10), bin_func=np.median, error_func="quantiles")

    plt.xlim(0, 100)
    plt.ylim(0, 200)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.250000/2.54, 5.000000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.124970, 0.226507, 0.245158, 0.648012])
plt.figure(1).axes[0].set_ylim(0.0, 220.0)
plt.figure(1).axes[0].set_yticklabels(["0", "50", "100", "150", "200"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[0].set_yticks([0.0, 50.0, 100.0, 150.0, 200.0])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.438800, 1.046685])
plt.figure(1).axes[0].texts[0].set_text("inlet")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_position([-0.343920, 1.046685])
plt.figure(1).axes[0].texts[1].set_text("a")
plt.figure(1).axes[0].texts[1].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("individual stiffness")
plt.figure(1).axes[1].set_position([0.419159, 0.226507, 0.245158, 0.648012])
plt.figure(1).axes[1].set_ylim(0.0, 220.0)
plt.figure(1).axes[1].set_yticklabels(["0", "50", "100", "150", "200"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[1].set_yticks([0.0, 50.0, 100.0, 150.0, 200.0])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.490337, 1.046685])
plt.figure(1).axes[1].texts[0].set_text("middle")
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[1].new
plt.figure(1).axes[1].texts[1].set_position([-0.099119, 1.046685])
plt.figure(1).axes[1].texts[1].set_text("b")
plt.figure(1).axes[1].texts[1].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[2].set_position([0.713349, 0.226507, 0.245158, 0.648012])
plt.figure(1).axes[2].set_ylim(0.0, 220.0)
plt.figure(1).axes[2].set_yticklabels(["0", "50", "100", "150", "200"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[2].set_yticks([0.0, 50.0, 100.0, 150.0, 200.0])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.493558, 1.046685])
plt.figure(1).axes[2].texts[0].set_text("outlet")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[1].new
plt.figure(1).axes[2].texts[1].set_position([-0.102340, 1.046685])
plt.figure(1).axes[2].texts[1].set_text("c")
plt.figure(1).axes[2].texts[1].set_weight("bold")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("radial position (µm)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3] + ".png", dpi=300)
plt.savefig(__file__[:-3] + ".pdf")
plt.show()
