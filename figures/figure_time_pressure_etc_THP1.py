import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new, bootstrap_match_hist
import numpy as np
import pylustrator

pylustrator.start()

data, config = load_all_data_new([
    fr"\\131.188.117.96\biophysDS\meroles\SPRING2021\24.03.2021_THP1_1.5%Alginate\**\*_evaluated_new.csv",
    fr"\\131.188.117.96\biophysDS\meroles\SPRING2021\24.03.2021_THP1_2%Alginate\**\*_evaluated_new.csv",
    fr"\\131.188.117.96\biophysDS\meroles\SPRING2021\25.03.2021_THP1_2.5%Alginate\**\*_evaluated_new.csv",
    fr"\\131.188.117.96\biophysDS\meroles\SPRING2021\30.03.2021_THP1_Ag2%\**\*_evaluated_new.csv",
    fr"\\131.188.117.96\biophysDS\meroles\SPRING2021\30.03.2021_THP1_Ag2%_3\**\*_evaluated_new.csv",
])


def plotMeasurementOfParameter(data, parameter, value, color=None):
    k_cell = data.groupby([parameter, "measurement_id"]).median()[value].reset_index()
    k_cell_mean = k_cell.groupby(parameter).mean()
    l = plt.errorbar(k_cell_mean.index, k_cell_mean.values, k_cell.groupby(parameter).sem().values[:, 0], capsize=3,
                     color=color, zorder=2)
    for measurement_id in k_cell.measurement_id.unique():
        meas_data = k_cell[k_cell.measurement_id == measurement_id]
        plt.plot(meas_data[parameter], meas_data[value], "o", ms=3, color=l[0].get_color(), alpha=0.5)
        plt.plot(meas_data[parameter], meas_data[value], "-", ms=3, color="gray", alpha=0.5)
        #plt.text(meas_data[parameter].values[0], meas_data[value].values[0], measurement_id)
    plt.xlabel(parameter)
    plt.ylabel(value)


def plot_pair(data, parameter):
    plt.axes([0.1, .2, 0.4, 0.5], label=f"{parameter}_alpha")
    plotMeasurementOfParameter(data, parameter, "w_k_cell", "C0")
    plt.axes([0.5, .2, 0.5, 0.5], label=f"{parameter}_k")
    plotMeasurementOfParameter(data, parameter, "w_alpha_cell", "C1")

data.pressure = data.pressure.round(3)
data.alginate = data.alginate.round(3)
print("pressurs", data.pressure.unique())
data = data[(data.pressure >= 0.9)*(data.pressure <= 3.1)]
print("pressurs", data.pressure.unique())
data_p3 = data[data.pressure == 2.0]
data_p3 = data_p3[data_p3.alginate == 2.0]
plot_pair(data_p3, "time")

data_alg2 = data[data.alginate.round(2) == 2.0]
plot_pair(data_alg2, "pressure")

data_alg_p2 = data[data.pressure.round(2) == 2.0]
#plot_pair(data_alg_p2, "alginate")

d = data_alg_p2
d["vel_grad_abs"] = np.abs(d.vel_grad)
d15 = d[d.alginate == 1.5]
d20 = d[d.alginate == 2.0]
d25 = d[d.alginate == 2.5]
d15_, d20_, d25_ = bootstrap_match_hist((d15, d20, d25), 10, 300, "vel_grad_abs")
plot_pair(pd.concat((d15_, d20_, d25_)), "alginate")

data_pos_p3 = data[data.alginate == 2.0]
data_pos_p3 = data_pos_p3[data_pos_p3.pressure == 2.0]
#plot_pair(data_pos_p3, "pos")

data_p3 = data[data.pressure == 2.0].copy()
data_p3 = data_p3[data_p3.alginate == 2.0]
bins = np.percentile(data_p3.area, [0, 33, 66, 100])
print("cell size, bins", bins)
data_p3["cell_size"] = "1small"
data_p3.loc[data_p3.area > bins[1], "cell_size"] = "2medium"
data_p3.loc[data_p3.area > bins[2], "cell_size"] = "3big"

plot_pair(data_p3, "cell_size")


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(16.980000/2.54, 5.910000/2.54, forward=True)
plt.figure(1).ax_dict["alginate_alpha"].set_xlim(1.3952195212681158, 2.6032812679138617)
plt.figure(1).ax_dict["alginate_alpha"].set_ylim(0.0, 190.0)
plt.figure(1).ax_dict["alginate_alpha"].set_yticks([0.0, 50.0, 100.0, 150.0, np.nan])
plt.figure(1).ax_dict["alginate_alpha"].set_xticklabels([])
plt.figure(1).ax_dict["alginate_alpha"].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["alginate_alpha"].set_position([0.527228, 0.597512, 0.173844, 0.329480])
plt.figure(1).ax_dict["alginate_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["alginate_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["alginate_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["alginate_alpha"].transAxes)  # id=plt.figure(1).ax_dict["alginate_alpha"].texts[0].new
plt.figure(1).ax_dict["alginate_alpha"].texts[0].set_position([-0.165224, 1.038176])
plt.figure(1).ax_dict["alginate_alpha"].texts[0].set_text("e")
plt.figure(1).ax_dict["alginate_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["alginate_alpha"].get_xaxis().get_label().set_text('')
plt.figure(1).ax_dict["alginate_alpha"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["alginate_k"].set_xlim(1.3952195212681158, 2.6032812679138617)
plt.figure(1).ax_dict["alginate_k"].set_ylim(0.0, 0.45)
plt.figure(1).ax_dict["alginate_k"].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
plt.figure(1).ax_dict["alginate_k"].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["alginate_k"].set_position([0.529005, 0.196666, 0.173844, 0.329480])
plt.figure(1).ax_dict["alginate_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["alginate_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["alginate_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["alginate_k"].transAxes)  # id=plt.figure(1).ax_dict["alginate_k"].texts[0].new
plt.figure(1).ax_dict["alginate_k"].texts[0].set_position([-0.165224, 0.974992])
plt.figure(1).ax_dict["alginate_k"].texts[0].set_text("f")
plt.figure(1).ax_dict["alginate_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["alginate_k"].get_xaxis().get_label().set_text("alginate %")
plt.figure(1).ax_dict["alginate_k"].get_yaxis().get_label().set_text('')
plt.figure(1).ax_dict["cell_size_alpha"].set_xlim(-0.14763368056372056, 2.1384352600208723)
plt.figure(1).ax_dict["cell_size_alpha"].set_ylim(0.0, 190.0)
plt.figure(1).ax_dict["cell_size_alpha"].set_xticks([0.0, 1.0, 2.0])
plt.figure(1).ax_dict["cell_size_alpha"].set_yticks([0.0, 50.0, 100.0, 150.0, np.nan])
plt.figure(1).ax_dict["cell_size_alpha"].set_xticklabels(["", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).ax_dict["cell_size_alpha"].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["cell_size_alpha"].set_position([0.733013, 0.597512, 0.173844, 0.329480])
plt.figure(1).ax_dict["cell_size_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["cell_size_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["cell_size_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cell_size_alpha"].transAxes)  # id=plt.figure(1).ax_dict["cell_size_alpha"].texts[0].new
plt.figure(1).ax_dict["cell_size_alpha"].texts[0].set_position([-0.128998, 1.038176])
plt.figure(1).ax_dict["cell_size_alpha"].texts[0].set_text("g")
plt.figure(1).ax_dict["cell_size_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["cell_size_alpha"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["cell_size_alpha"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["cell_size_k"].set_xlim(-0.12326438572207858, 2.1488649996704448)
plt.figure(1).ax_dict["cell_size_k"].set_ylim(0.0, 0.45)
plt.figure(1).ax_dict["cell_size_k"].set_xticks([0.0, 1.0, 2.0])
plt.figure(1).ax_dict["cell_size_k"].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
plt.figure(1).ax_dict["cell_size_k"].set_xticklabels(["small", "medium", "big"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).ax_dict["cell_size_k"].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["cell_size_k"].set_position([0.734791, 0.198965, 0.173844, 0.329480])
plt.figure(1).ax_dict["cell_size_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["cell_size_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["cell_size_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cell_size_k"].transAxes)  # id=plt.figure(1).ax_dict["cell_size_k"].texts[0].new
plt.figure(1).ax_dict["cell_size_k"].texts[0].set_position([-0.128998, 0.968015])
plt.figure(1).ax_dict["cell_size_k"].texts[0].set_text("h")
plt.figure(1).ax_dict["cell_size_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["cell_size_k"].get_xaxis().get_label().set_text("cell size")
plt.figure(1).ax_dict["cell_size_k"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["pressure_alpha"].set_xlim(0.5, 3.5)
plt.figure(1).ax_dict["pressure_alpha"].set_ylim(0.0, 190.0)
plt.figure(1).ax_dict["pressure_alpha"].set_yticks([0.0, 50.0, 100.0, 150.0, np.nan])
plt.figure(1).ax_dict["pressure_alpha"].set_xticklabels([])
plt.figure(1).ax_dict["pressure_alpha"].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["pressure_alpha"].set_position([0.321443, 0.594697, 0.173844, 0.329480])
plt.figure(1).ax_dict["pressure_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pressure_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pressure_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["pressure_alpha"].transAxes)  # id=plt.figure(1).ax_dict["pressure_alpha"].texts[0].new
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_position([-0.257195, 1.038176])
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_text("c")
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["pressure_alpha"].get_xaxis().get_label().set_text('')
plt.figure(1).ax_dict["pressure_alpha"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["pressure_k"].set_xlim(0.5, 3.5)
plt.figure(1).ax_dict["pressure_k"].set_ylim(0.0, 0.45)
plt.figure(1).ax_dict["pressure_k"].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
plt.figure(1).ax_dict["pressure_k"].set_yticklabels(["", "", "", "", ""], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["pressure_k"].set_position([0.321442, 0.193851, 0.173844, 0.329480])
plt.figure(1).ax_dict["pressure_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["pressure_k"].transAxes)  # id=plt.figure(1).ax_dict["pressure_k"].texts[0].new
plt.figure(1).ax_dict["pressure_k"].texts[0].set_position([-0.257195, 0.974992])
plt.figure(1).ax_dict["pressure_k"].texts[0].set_text("d")
plt.figure(1).ax_dict["pressure_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["pressure_k"].get_xaxis().get_label().set_text("pressure (bar)")
plt.figure(1).ax_dict["pressure_k"].get_yaxis().get_label().set_text("")
plt.figure(1).ax_dict["time_alpha"].set_xlim(0.0, 67.0)
plt.figure(1).ax_dict["time_alpha"].set_ylim(0.0, 190.0)
plt.figure(1).ax_dict["time_alpha"].set_yticks([0.0, 50.0, 100.0, 150.0])
plt.figure(1).ax_dict["time_alpha"].set_xticklabels([])
plt.figure(1).ax_dict["time_alpha"].set_yticklabels(["0", "50", "100", "150"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["time_alpha"].legend(handletextpad=0.7999999999999999, fontsize=8.0, title_fontsize=7.0)
plt.figure(1).ax_dict["time_alpha"].set_position([0.085637, 0.594181, 0.208913, 0.329996])
plt.figure(1).ax_dict["time_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["time_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["time_alpha"].get_legend().set_visible(False)
plt.figure(1).ax_dict["time_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).ax_dict["time_alpha"].texts[0].set_position([-0.380362, 0.894436])
plt.figure(1).ax_dict["time_alpha"].texts[0].set_text("b")
plt.figure(1).ax_dict["time_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["time_alpha"].get_xaxis().get_label().set_text("")
plt.figure(1).ax_dict["time_alpha"].get_yaxis().get_label().set_text("stiffness (Pa)")
plt.figure(1).ax_dict["time_k"].set_xlim(0.0, 67.0)
plt.figure(1).ax_dict["time_k"].set_ylim(0.0, 0.45)
plt.figure(1).ax_dict["time_k"].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
plt.figure(1).ax_dict["time_k"].set_yticklabels(["0.0", "0.1", "0.2", "0.3", "0.4"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="right")
plt.figure(1).ax_dict["time_k"].set_position([0.085636, 0.196150, 0.208913, 0.329996])
plt.figure(1).ax_dict["time_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["time_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["time_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[1].transAxes)  # id=plt.figure(1).axes[1].texts[0].new
plt.figure(1).ax_dict["time_k"].texts[0].set_position([-0.258550, 1.038117])
plt.figure(1).ax_dict["time_k"].texts[0].set_text("a")
plt.figure(1).ax_dict["time_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["time_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["time_k"].transAxes)  # id=plt.figure(1).ax_dict["time_k"].texts[1].new
plt.figure(1).ax_dict["time_k"].texts[1].set_position([-0.258545, 0.966501])
plt.figure(1).ax_dict["time_k"].texts[1].set_text("b")
plt.figure(1).ax_dict["time_k"].texts[1].set_weight("bold")
plt.figure(1).ax_dict["time_k"].get_xaxis().get_label().set_text("time (min)")
plt.figure(1).ax_dict["time_k"].get_yaxis().get_label().set_text("alpha")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()
