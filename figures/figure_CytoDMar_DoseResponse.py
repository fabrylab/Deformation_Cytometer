import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new, bootstrap_match_hist
import numpy as np
import pylustrator

pylustrator.start()

def get_mode_stats(x):
    from scipy import stats
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    x = np.array(x)
    x = x[~np.isnan(x)&(x>0)]
    def get_mode(x):
        kde = stats.gaussian_kde(x)
        return x[np.argmax(kde(x))]
    mode = get_mode(x)
    #err = bootstrap_error(x, get_mode, repetitions=2)
    return mode


def get_mode_stats_log(x):
    from scipy import stats
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    x = np.array(x)
    x = x[~np.isnan(x)&(x>0)]
    def get_mode(x):
        x = np.log(x)
        kde = stats.gaussian_kde(x)
        return np.exp(x[np.argmax(kde(x))])
    mode = get_mode(x)
    #err = bootstrap_error(x, get_mode, repetitions=2)
    return mode

def get_mode_stats_err(x):
    from scipy import stats
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    x = np.array(x)
    x = x[~np.isnan(x)&(x>0)]
    def get_mode(x):
        x = np.log(x)
        kde = stats.gaussian_kde(x)
        return np.exp(x[np.argmax(kde(x))])
    #mode = get_mode(x)
    err = bootstrap_error(x, get_mode, repetitions=2)
    return err

def plotMeasurementOfParameter(data, parameter, value, agg="mean", color=None, label=None):
    agg_func = {"mean": np.mean, "median": np.median, "mode": get_mode_stats, "logmode": get_mode_stats_log}[agg]

    agg = data.groupby([parameter, "measurement_id"])[value].agg(agg_func).reset_index()
    #agg_err = data.groupby([parameter, "measurement_id"])[value].agg(get_mode_stats_err).reset_index()
    #agg = data.groupby([parameter, "measurement_id"])[value].mean().reset_index()
    #agg_err = data.groupby([parameter, "measurement_id"])[value].sem().reset_index()
    agg_mean = agg.groupby(parameter)[value].mean()
    agg_err_mean = agg.groupby(parameter)[value].sem()#agg(lambda x: 0.5*np.sqrt((x**2).sum()))
    l = plt.errorbar(agg_mean.index, agg_mean.values, agg_err_mean.values, capsize=3, color=color, zorder=2, label=label)
    for measurement_id, meas_data in agg.groupby("measurement_id"):
        plt.plot(meas_data[parameter], meas_data[value], "o", ms=3, color=l[0].get_color(), alpha=0.5)
        plt.plot(meas_data[parameter], meas_data[value], "-", ms=3, color="gray", alpha=0.5)
        #plt.text(meas_data[parameter].values[0], meas_data[value].values[0], measurement_id)
    plt.xlabel(parameter)
    plt.ylabel(value)


def plot_pair(data, parameter, color=["C0", "C1"], label=None, scaling=None):
    plt.axes([0.05, .2, 0.4, 0.5], label=f"{parameter}_alpha")
    plotMeasurementOfParameter(data, parameter, "w_k_cell", agg="logmode", color=color[0], label=label)
    plt.ylim(bottom=0)
    if scaling == "semilogx":
        plt.semilogx()
    plt.axes([0.55, .2, 0.4, 0.5], label=f"{parameter}_k")
    plotMeasurementOfParameter(data, parameter, "w_alpha_cell", agg="mode", color=color[1], label=label)
    plt.ylim(bottom=0)
    if scaling == "semilogx":
        plt.semilogx()


data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\24.03.2021_THP1_2%Alginate\**\*_evaluated_new.csv",
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\30.03.2021_THP1_Ag2%\**\*_evaluated_new.csv",
#r"\\131.188.117.96\biophysDS\meroles\SPRING2021\30.03.2021_THP1_Ag2%_2\**\*_evaluated_new.csv",
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\30.03.2021_THP1_Ag2%_3\**\*_evaluated_new.csv",

r"\\131.188.117.96\biophysDS\meroles\SPRING2021\22.04.2021_THP1_CytoD_*\**\*_evaluated_new.csv",
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\21.04.2021_THP1_Cyto*\**\*_evaluated_new.csv",
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\21.04.2021_THP1_*Cyto\**\*_evaluated_new.csv",
r"\\131.188.117.96\biophysDS\meroles\SPRING2021\23.04.2021_THP1_Cyto*M*\**\*_evaluated_new.csv",
], pressure=2)
# filter only the 60min data
data = data[data.time == 60]
# print the table to chech that all folders where loaded in correctly
print(data.groupby(["measurement_id", "datetime"])["cytoD"].mean().sort_values("cytoD"))
# convert cytoD form nM to µM
data.cytoD /= 1e3
# fake the 0 value to 0.001 so that it can be displayed in the logplot
data.loc[data.cytoD.isnull(), 'cytoD'] = 0.001
# plot the data
plot_pair(data, "cytoD", ["C3", "C3"], scaling="semilogx")

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(11.990000/2.54, 4.990000/2.54, forward=True)
plt.figure(1).ax_dict["cytoD_alpha"].set_xlim(0.001, 14.12537544622754)
plt.figure(1).ax_dict["cytoD_alpha"].set_xticks([0.001, 0.01, 0.1, 1.0, 10.0])
plt.figure(1).ax_dict["cytoD_alpha"].set_xticklabels(["0", "0.01", "0.1", "1", "10"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).ax_dict["cytoD_alpha"].set_position([0.113746, 0.288732, 0.382668, 0.591726])
plt.figure(1).ax_dict["cytoD_alpha"].set_xticklabels(["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""], minor=True)
plt.figure(1).ax_dict["cytoD_alpha"].set_xticks([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], minor=True)
plt.figure(1).ax_dict["cytoD_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["cytoD_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["cytoD_alpha"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cytoD_alpha"].transAxes)  # id=plt.figure(1).ax_dict["cytoD_alpha"].texts[0].new
plt.figure(1).ax_dict["cytoD_alpha"].texts[0].set_position([-0.130531, 1.025424])
plt.figure(1).ax_dict["cytoD_alpha"].texts[0].set_text("a")
plt.figure(1).ax_dict["cytoD_alpha"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["cytoD_alpha"].get_xaxis().get_label().set_text("cytoD (µM)")
plt.figure(1).ax_dict["cytoD_alpha"].get_yaxis().get_label().set_text("stiffness (Pa)")
plt.figure(1).ax_dict["cytoD_k"].set_xlim(0.001, 14.12537544622754)
plt.figure(1).ax_dict["cytoD_k"].set_xticks([0.001, 0.01, 0.1, 1.0, 10.0])
plt.figure(1).ax_dict["cytoD_k"].set_xticklabels(["0", "0.01", "0.1", "1", "10"], fontsize=10.0, fontweight="normal", color="black", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).ax_dict["cytoD_k"].set_position([0.592081, 0.288732, 0.382668, 0.591726])
plt.figure(1).ax_dict["cytoD_k"].set_xticklabels(["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""], minor=True)
plt.figure(1).ax_dict["cytoD_k"].set_xticks([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], minor=True)
plt.figure(1).ax_dict["cytoD_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["cytoD_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["cytoD_k"].text(0.5, 0.5, 'New Text', transform=plt.figure(1).ax_dict["cytoD_k"].transAxes)  # id=plt.figure(1).ax_dict["cytoD_k"].texts[0].new
plt.figure(1).ax_dict["cytoD_k"].texts[0].set_position([-0.119469, 1.025424])
plt.figure(1).ax_dict["cytoD_k"].texts[0].set_text("b")
plt.figure(1).ax_dict["cytoD_k"].texts[0].set_weight("bold")
plt.figure(1).ax_dict["cytoD_k"].get_xaxis().get_label().set_text("cytoD (µM)")
plt.figure(1).ax_dict["cytoD_k"].get_yaxis().get_label().set_text("alpha")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()
