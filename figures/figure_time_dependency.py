# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import plotDensityScatter, plotStressStrainFit, load_all_data, all_plots_same_limits

import pylustrator
pylustrator.start()

rows = 3
cols = 5

for index, time in enumerate(range(1, 6, 1)):
    plt.subplot(rows, cols, index+1)
    data, config = load_all_data([
        rf"\\131.188.117.96\biophysDS\meroles\2020.07.07.Control.sync\Control2\T{time*10}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\meroles\2020.06.26\Control2\T{time*10}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\meroles\2020.06.26\Control1\T{time*10}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\meroles\2020.07.09_control_sync_hepes\Control3\T{time*10}\*_result.txt",
    ], repetition=2)

    plotDensityScatter(data.stress, data.strain)
    plotStressStrainFit(data, config)

for index, time in enumerate(range(2, 11, 2)):
    plt.subplot(rows, cols, cols+index+1)
    data, config = load_all_data([
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_10_alginate2%_K562_0%FCS_time\{time}\*_result.txt",
#        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\07_07_2020_alginate2%_rpmi_no_fcs_time\{time}\*_result.txt",
    ], repetition=2)

    plotDensityScatter(data.stress, data.strain)
    plotStressStrainFit(data, config)

for index, time in enumerate(range(2, 11, 2)):
    plt.subplot(rows, cols, cols*2+index+1)
    data, config = load_all_data([
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_may\2020_05_22_alginateDMEM2%\{time}\*_result.txt",
    ], repetition=1)

    plotDensityScatter(data.stress, data.strain)
    plotStressStrainFit(data, config)

all_plots_same_limits()

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
plt.figure(1).axes[0].set_position([0.110316, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[0].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[0].set_xticklabels(["", ""])
plt.figure(1).axes[0].set_xticks([0.0, 100.0])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.509148, 0.891045])
plt.figure(1).axes[0].texts[0].set_text("10 min")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("THP1\nstrain")
plt.figure(1).axes[1].set_position([0.289717, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[1].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[1].set_xticklabels(["", ""])
plt.figure(1).axes[1].set_xticks([0.0, 100.0])
plt.figure(1).axes[1].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[1].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[1].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).axes[1].texts[0].set_ha("center")
plt.figure(1).axes[1].texts[0].set_position([0.545742, 0.891045])
plt.figure(1).axes[1].texts[0].set_text("20 min")
plt.figure(1).axes[2].set_position([0.469119, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[2].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[2].set_xticklabels(["", ""])
plt.figure(1).axes[2].set_xticks([0.0, 100.0])
plt.figure(1).axes[2].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[2].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[2].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[2].spines['right'].set_visible(False)
plt.figure(1).axes[2].spines['top'].set_visible(False)
plt.figure(1).axes[2].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[2].transAxes)  # id=plt.figure(1).axes[2].texts[0].new
plt.figure(1).axes[2].texts[0].set_ha("center")
plt.figure(1).axes[2].texts[0].set_position([0.504574, 0.891045])
plt.figure(1).axes[2].texts[0].set_text("30 min")
plt.figure(1).axes[3].set_position([0.648521, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[3].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[3].set_xticklabels(["", ""])
plt.figure(1).axes[3].set_xticks([0.0, 100.0])
plt.figure(1).axes[3].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[3].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[3].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[3].spines['right'].set_visible(False)
plt.figure(1).axes[3].spines['top'].set_visible(False)
plt.figure(1).axes[3].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[3].transAxes)  # id=plt.figure(1).axes[3].texts[0].new
plt.figure(1).axes[3].texts[0].set_ha("center")
plt.figure(1).axes[3].texts[0].set_position([0.518297, 0.891045])
plt.figure(1).axes[3].texts[0].set_text("50 min")
plt.figure(1).axes[4].set_position([0.827922, 0.716785, 0.160511, 0.272740])
plt.figure(1).axes[4].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[4].set_xticklabels(["", ""])
plt.figure(1).axes[4].set_xticks([0.0, 100.0])
plt.figure(1).axes[4].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[4].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[4].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[4].spines['right'].set_visible(False)
plt.figure(1).axes[4].spines['top'].set_visible(False)
plt.figure(1).axes[4].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[4].transAxes)  # id=plt.figure(1).axes[4].texts[0].new
plt.figure(1).axes[4].texts[0].set_ha("center")
plt.figure(1).axes[4].texts[0].set_position([0.532020, 0.891045])
plt.figure(1).axes[4].texts[0].set_text("60 min")
plt.figure(1).axes[5].set_position([0.110316, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[5].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[5].set_xticklabels(["", ""])
plt.figure(1).axes[5].set_xticks([0.0, 100.0])
plt.figure(1).axes[5].spines['right'].set_visible(False)
plt.figure(1).axes[5].spines['top'].set_visible(False)
plt.figure(1).axes[5].get_yaxis().get_label().set_text("K562\nstrain")
plt.figure(1).axes[6].set_position([0.289717, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[6].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[6].set_xticklabels(["", ""])
plt.figure(1).axes[6].set_xticks([0.0, 100.0])
plt.figure(1).axes[6].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[6].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[6].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[6].spines['right'].set_visible(False)
plt.figure(1).axes[6].spines['top'].set_visible(False)
plt.figure(1).axes[7].set_position([0.469119, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[7].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[7].set_xticklabels(["", ""])
plt.figure(1).axes[7].set_xticks([0.0, 100.0])
plt.figure(1).axes[7].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[7].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[7].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[7].spines['right'].set_visible(False)
plt.figure(1).axes[7].spines['top'].set_visible(False)
plt.figure(1).axes[8].set_position([0.648521, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[8].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[8].set_xticklabels(["", ""])
plt.figure(1).axes[8].set_xticks([0.0, 100.0])
plt.figure(1).axes[8].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[8].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[8].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[8].spines['right'].set_visible(False)
plt.figure(1).axes[8].spines['top'].set_visible(False)
plt.figure(1).axes[9].set_position([0.827922, 0.407032, 0.160511, 0.272740])
plt.figure(1).axes[9].set_xlim(-9.25752281870522, 193.99024490625322)
plt.figure(1).axes[9].set_xticklabels(["", ""])
plt.figure(1).axes[9].set_xticks([0.0, 100.0])
plt.figure(1).axes[9].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[9].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[9].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[9].spines['right'].set_visible(False)
plt.figure(1).axes[9].spines['top'].set_visible(False)
plt.figure(1).axes[10].set_position([0.110316, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[10].spines['right'].set_visible(False)
plt.figure(1).axes[10].spines['top'].set_visible(False)
plt.figure(1).axes[10].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[10].get_yaxis().get_label().set_text("NIH 3T3\nstrain")
plt.figure(1).axes[11].set_position([0.289717, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[11].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[11].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[11].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[11].spines['right'].set_visible(False)
plt.figure(1).axes[11].spines['top'].set_visible(False)
plt.figure(1).axes[11].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[12].set_position([0.469119, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[12].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[12].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[12].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[12].spines['right'].set_visible(False)
plt.figure(1).axes[12].spines['top'].set_visible(False)
plt.figure(1).axes[12].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[13].set_position([0.648521, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[13].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[13].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[13].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[13].spines['right'].set_visible(False)
plt.figure(1).axes[13].spines['top'].set_visible(False)
plt.figure(1).axes[13].get_xaxis().get_label().set_text("stress (Pa)")
plt.figure(1).axes[14].set_position([0.827922, 0.097280, 0.160511, 0.272740])
plt.figure(1).axes[14].set_ylim(-0.10347851610695331, 1.6109644391156153)
plt.figure(1).axes[14].set_yticklabels(["", "", "", ""])
plt.figure(1).axes[14].set_yticks([0.0, 0.5, 1.0, 1.5])
plt.figure(1).axes[14].spines['right'].set_visible(False)
plt.figure(1).axes[14].spines['top'].set_visible(False)
plt.figure(1).axes[14].get_xaxis().get_label().set_text("stress (Pa)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png")
plt.savefig(__file__[:-3]+".pdf")
plt.show()


