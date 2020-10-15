# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from scripts.helper_functions import plotDensityScatter
from scripts.helper_functions import load_all_data
import numpy as np

import pylustrator
pylustrator.start()

numbers = []
for index, pressure in enumerate([1, 2, 3]):
    ax = plt.subplot(1, 3, index+1)

    #data, config = load_all_data(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data"+
    #                             r"\microscope4\2020_may\2020_05_22_alginateDMEM2%\[0-9]\*_result.txt", pressure=pressure)
    data, config = load_all_data([
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_2\inlet\[0-9]\*_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\[0-9]\*_result.txt",
#        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\[0-9]\*_result.txt",
#        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\[0-9]\*_result.txt",
#        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_27_alginate2%_dmem_NIH_time_1\[0-9]\*_result.txt",
    ], pressure=pressure)

    plotDensityScatter(data.rp, data.strain)
    numbers.append(len(data.rp))
    print(config)

print("numbers", numbers)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.260000/2.54, 4.390000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.115765, 0.257929, 0.268860, 0.717020])
plt.figure(1).axes[0].set_xlim(-90.0, 90.0)
plt.figure(1).axes[0].set_xticklabels(["-75", "0", "75"])
plt.figure(1).axes[0].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[0].set_xticks([np.nan], minor=True)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.509148, 0.891045])
plt.figure(1).axes[0].texts[0].set_text("1 bar")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_position([-0.232258, 0.955913])
plt.figure(1).axes[0].texts[1].set_text("a")
plt.figure(1).axes[0].texts[1].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("NIH 3T3\nstrain")
plt.figure(1).axes[1].set_position([0.416266, 0.257929, 0.268860, 0.717020])
plt.figure(1).axes[1].set_xlim(-90.0, 90.0)
plt.figure(1).axes[1].set_xticklabels(["-75", "0", "75"])
plt.figure(1).axes[1].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[1].set_xticks([np.nan], minor=True)
plt.figure(1).axes[1].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[1].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[1].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].xaxis.labelpad = 3.716691
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.545742, 0.891045])
plt.figure(1).axes[1].texts[0].set_text("2 bar")
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[1].new
plt.figure(1).axes[1].texts[1].set_position([-0.127650, 0.955913])
plt.figure(1).axes[1].texts[1].set_text("b")
plt.figure(1).axes[1].texts[1].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[2].set_position([0.716769, 0.257929, 0.268860, 0.717020])
plt.figure(1).axes[2].set_xlim(-90.0, 90.0)
plt.figure(1).axes[2].set_xticklabels(["-75", "0", "75"])
plt.figure(1).axes[2].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[2].set_xticks([np.nan], minor=True)
plt.figure(1).axes[2].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[2].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[2].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.504574, 0.891045])
plt.figure(1).axes[2].texts[0].set_text("3 bar")
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[1].new
plt.figure(1).axes[2].texts[1].set_position([-0.110215, 0.955913])
plt.figure(1).axes[2].texts[1].set_text("c")
plt.figure(1).axes[2].texts[1].set_weight("bold")
plt.figure(1).axes[2].get_xaxis().get_label().set_text("radial position (µm)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


