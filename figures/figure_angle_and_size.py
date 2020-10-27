# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, load_all_data
import numpy as np

import pylustrator
pylustrator.start()

for index, pressure in enumerate([1]):
    ax = plt.subplot(1, 3, index+1)

    data, config = load_all_data([
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_21_alginate2%_dmem_NIH_time_2\[0-9]\*_result.txt",
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\[0-9]\*_result.txt",
        r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_27_alginate2%_dmem_NIH_time_1\[0-9]\*_result.txt",
    ], pressure=pressure)

    plt.subplot(1, 2, 1)
    plotDensityScatter(data.rp, data.angle)

    plt.subplot(1, 2, 2)
    plotDensityScatter(data.rp, np.sqrt(data.area / np.pi))


#plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.000000/2.54, 4.370000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.118367, 0.241321, 0.356764, 0.716233])
plt.figure(1).axes[0].set_xlim(-90.0, 90.0)
plt.figure(1).axes[0].set_xticklabels(["-75", "0", "75"])
plt.figure(1).axes[0].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[0].set_xticks([np.nan], minor=True)
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[0].set_position([-0.232258, 0.955913])
plt.figure(1).axes[0].texts[0].set_text("a")
plt.figure(1).axes[0].texts[0].set_weight("bold")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("angle (deg)")
plt.figure(1).axes[1].set_position([0.618713, 0.241321, 0.356764, 0.716233])
plt.figure(1).axes[1].set_xlim(-90.0, 90.0)
plt.figure(1).axes[1].set_xticklabels(["-75", "0", "75"])
plt.figure(1).axes[1].set_xticks([-75.0, 0.0, 75.0])
plt.figure(1).axes[1].set_xticks([np.nan], minor=True)
plt.figure(1).axes[1].set_ylim(0.0, 30.0)
plt.figure(1).axes[1].set_yticklabels(["0", "5", "10", "15", "20", "25", "30"])
plt.figure(1).axes[1].set_yticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].xaxis.labelpad = 3.716691
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[1].new
plt.figure(1).axes[1].texts[0].set_position([-0.354809, 0.955913])
plt.figure(1).axes[1].texts[0].set_text("b")
plt.figure(1).axes[1].texts[0].set_weight("bold")
plt.figure(1).axes[1].get_xaxis().get_label().set_text("radial position (µm)")
plt.figure(1).axes[1].get_yaxis().get_label().set_text("undeformed\ncell diameter (µm)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()


