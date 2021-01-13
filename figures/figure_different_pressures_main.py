# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data, plotBinnedData
import numpy as np

import pylustrator
pylustrator.start()

numbers = []
data_list = []
config_list = []
for index, pressure in enumerate([1, 2, 3]):
    data, config = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
#     r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\september_2020\2020_09_16_alginate2%_NIH_tanktreading\*\*_result.txt",
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

    plotDensityScatter(data.rp, data.strain)

    ax = plt.subplot(2, 3, 3+index+1)
    plotDensityScatter(data.stress, data.strain)
    for i in range(3):
        if i != index:
            plotBinnedData(data_list[i].stress, data_list[i].strain, ranges[i], error_func="quantiles", alpha=0.5, mec="none", mfc=colors[i])
    plotBinnedData(data.stress, data.strain, ranges[index], error_func="quantiles", mfc=colors[index])
    numbers.append(len(data.rp))
    print(config)

print("numbers", numbers)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 10.890000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(-90.22751034357464, 89.45098813781398)
plt.figure(1).axes[0].set_ylim(-0.05093250462193348, 1.156151079632439)
plt.figure(1).axes[0].set_position([0.111408, 0.621312, 0.269192, 0.340314])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.500000, 0.910714])
plt.figure(1).axes[0].texts[0].set_text("1 bar")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_position([-0.283595, 0.910714])
plt.figure(1).axes[0].texts[1].set_text("a")
plt.figure(1).axes[0].texts[1].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("cell strain $\epsilon$")
plt.figure(1).axes[1].set_xlim(-7.188368001375647, 266.02832381320513)
plt.figure(1).axes[1].set_ylim(-0.05093250462193348, 1.156151079632439)
plt.figure(1).axes[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.figure(1).axes[1].set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[1].set_position([0.111408, 0.151153, 0.269192, 0.340314])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_position([-0.283595, 0.918800])
plt.figure(1).axes[1].texts[0].set_text("d")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("fluid shear stress $\sigma$ (Pa)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("cell strain $\epsilon$")
plt.figure(1).axes[2].set_xlim(-90.22751034357464, 89.45098813781398)
plt.figure(1).axes[2].set_ylim(-0.05093250462193348, 1.156151079632439)
plt.figure(1).axes[2].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.figure(1).axes[2].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[2].set_position([0.415839, 0.621312, 0.269192, 0.340314])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.500000, 0.910714])
plt.figure(1).axes[2].texts[0].set_text("2 bar")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[1].new
plt.figure(1).axes[2].texts[1].set_position([-0.115267, 0.910714])
plt.figure(1).axes[2].texts[1].set_text("b")
plt.figure(1).axes[2].texts[1].set_weight("bold")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[3].set_xlim(-7.188368001375647, 266.02832381320513)
plt.figure(1).axes[3].set_ylim(-0.05093250462193348, 1.156151079632439)
plt.figure(1).axes[3].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.figure(1).axes[3].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[3].set_position([0.415839, 0.151153, 0.269192, 0.340314])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_position([-0.115267, 0.918800])
plt.figure(1).axes[3].texts[0].set_text("e")
plt.figure(1).axes[3].texts[0].set_weight("bold")
plt.figure(1).axes[3].get_xaxis().get_label().set_text("fluid shear stress $\sigma$ (Pa)")
plt.figure(1).axes[4].set_xlim(-90.22751034357464, 89.45098813781398)
plt.figure(1).axes[4].set_ylim(-0.05093250462193348, 1.156151079632439)
plt.figure(1).axes[4].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.figure(1).axes[4].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[4].set_position([0.720270, 0.621312, 0.269192, 0.340314])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_ha("center")
plt.figure(1).axes[4].texts[0].set_position([0.500000, 0.910714])
plt.figure(1).axes[4].texts[0].set_text("3 bar")
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[1].new
plt.figure(1).axes[4].texts[1].set_position([-0.097854, 0.910714])
plt.figure(1).axes[4].texts[1].set_text("c")
plt.figure(1).axes[4].texts[1].set_weight("bold")
plt.figure(1).axes[4].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[5].set_xlim(-7.188368001375647, 266.02832381320513)
plt.figure(1).axes[5].set_ylim(-0.05093250462193348, 1.156151079632439)
plt.figure(1).axes[5].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.figure(1).axes[5].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).axes[5].set_position([0.720270, 0.151153, 0.269192, 0.340314])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[5].transAxes)  # id=plt.figure(1).axes[5].texts[0].new
plt.figure(1).axes[5].texts[0].set_position([-0.097854, 0.918800])
plt.figure(1).axes[5].texts[0].set_text("f")
plt.figure(1).axes[5].texts[0].set_weight("bold")
plt.figure(1).axes[5].get_xaxis().get_label().set_text("fluid shear stress $\sigma$ (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


