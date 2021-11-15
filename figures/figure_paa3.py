import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new, bootstrap_match_hist, bootstrap_error, plotDensityScatter
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

def getFitParameters(data):
    from scipy.special import gamma
    def fit(omega, k, alpha):
        omega = np.array(omega)
        G = k * (1j * omega) ** alpha * gamma(1 - alpha)
        return np.real(G), np.imag(G)

    def cost(p):
        Gp1, Gp2 = fit(data.omega_weissenberg, *p)
        return np.sum(np.abs(np.log10(data.w_Gp1) - np.log10(Gp1))) + np.sum(
            np.abs(np.log10(data.w_Gp2) - np.log10(Gp2)))

    from scipy.optimize import minimize
    res = minimize(cost, [np.median(data.w_k_cell), np.mean(data.w_alpha_cell)], bounds=([0, np.inf], [0, 1]),
                   options={"disp": False})

    return res.x


def plotMeasurementOfParameter(data, parameter, value, agg="mean", color=None, label=None):
    agg_func = {"mean": np.mean, "median": np.median, "mode": get_mode_stats, "logmode": get_mode_stats_log}[agg]
    agg_func_err = {"mean": lambda x: np.std(x)/len(x), "median": lambda x: bootstrap_error(x, np.median),
                    "mode": lambda x: bootstrap_error(x, get_mode_stats, repetitions=2), "logmode": lambda x: bootstrap_error(x, get_mode_stats_log, repetitions=2)}[agg]

    if 1:
        #agg = data.groupby([parameter, "measurement_id"])[value].agg(agg_func).reset_index()
        # agg_err = data.groupby([parameter, "measurement_id"])[value].agg(get_mode_stats_err).reset_index()
        # agg = data.groupby([parameter, "measurement_id"])[value].mean().reset_index()
        # agg_err = data.groupby([parameter, "measurement_id"])[value].sem().reset_index()
        agg_mean = data.groupby(parameter)[value].agg(agg_func)
        agg_err_mean = data.groupby(parameter)[value].agg(agg_func_err)  # agg(lambda x: 0.5*np.sqrt((x**2).sum()))
        l = plt.errorbar(agg_mean.index, agg_mean.values, agg_err_mean.values, capsize=3, color=color, zorder=2,
                         label=label)
    else:
        agg = data.groupby([parameter, "measurement_id"])[value].agg(agg_func).reset_index()
        #agg_err = data.groupby([parameter, "measurement_id"])[value].agg(get_mode_stats_err).reset_index()
        #agg = data.groupby([parameter, "measurement_id"])[value].mean().reset_index()
        #agg_err = data.groupby([parameter, "measurement_id"])[value].sem().reset_index()
        agg_mean = agg.groupby(parameter)[value].mean()
        agg_err_mean = agg.groupby(parameter)[value].sem()#agg(lambda x: 0.5*np.sqrt((x**2).sum()))
        l = plt.errorbar(agg_mean.index, agg_mean.values, agg_err_mean.values, capsize=3, color=color, zorder=2, label=label, fmt="o-")
        for measurement_id, meas_data in agg.groupby("measurement_id"):
            plt.plot(meas_data[parameter], meas_data[value], "o", ms=3, color=l[0].get_color(), alpha=0.5)
            plt.plot(meas_data[parameter], meas_data[value], "-", ms=3, color="gray", alpha=0.5)
            #plt.text(meas_data[parameter].values[0], meas_data[value].values[0], measurement_id)
    plt.xlabel(parameter)
    plt.ylabel(value)


def plot_pair(data, parameter, color=["C0", "C1"], label=None):
    plt.axes([0.1, .2, 0.4, 0.5], label=f"{parameter}_alpha")
    plotMeasurementOfParameter(data, parameter, "w_k_cell", agg="median", color=color[0], label=label)
    plt.ylim(bottom=0)
    plt.axes([0.5, .2, 0.5, 0.5], label=f"{parameter}_k")
    plotMeasurementOfParameter(data, parameter, "w_alpha_cell", agg="mean", color=color[1], label=label)
    plt.ylim(bottom=0)

rep = 0
def plot_stiffness(data, parameter, color=["C0", "C1"], label=None, index=0):
    global rep
    plt.subplot(3, 2, 2)
    plt.title(label)
    agg_func = {"mean": np.mean, "median": np.median, "mode": get_mode_stats, "logmode": get_mode_stats_log}["logmode"]
    agg_func_err = {"mean": lambda x: np.std(x) / len(x), "median": lambda x: bootstrap_error(x, np.median),
                    "mode": lambda x: bootstrap_error(x, get_mode_stats, repetitions=2),
                    "logmode": lambda x: bootstrap_error(x, get_mode_stats_log, repetitions=2)}["logmode"]

    parameter = "pressure"
    value = "w_alpha_cell"
    value2 = "strain"
    agg_mean = data.groupby(parameter)[value].agg(agg_func)
    agg_mean2 = data.groupby(parameter)[value2].agg(agg_func)
    agg_err_mean = data.groupby(parameter)[value].agg(agg_func_err)  # agg(lambda x: 0.5*np.sqrt((x**2).sum()))
    l = plt.errorbar(agg_mean2.values, agg_mean.values, agg_err_mean.values, capsize=3, color=color[0], zorder=2,
                     label=label)

    plt.subplot(3, 2, 1)
    #plotDensityScatter(data.strain, data.w_alpha_cell)
    plt.title(label)

    agg_func = {"mean": np.mean, "median": np.median, "mode": get_mode_stats, "logmode": get_mode_stats_log}["logmode"]
    agg_func_err = {"mean": lambda x: np.std(x) / len(x), "median": lambda x: bootstrap_error(x, np.median),
                    "mode": lambda x: bootstrap_error(x, get_mode_stats, repetitions=2),
                    "logmode": lambda x: bootstrap_error(x, get_mode_stats_log, repetitions=2)}["logmode"]

    parameter = "pressure"
    value = "w_k_cell"
    value2 = "strain"
    agg_mean = data.groupby(parameter)[value].agg(agg_func)
    agg_mean2 = data.groupby(parameter)[value2].agg(agg_func)
    agg_err_mean = data.groupby(parameter)[value].agg(agg_func_err)  # agg(lambda x: 0.5*np.sqrt((x**2).sum()))
    l = plt.errorbar(agg_mean2.values, agg_mean.values, agg_err_mean.values, capsize=3, color=color[0], zorder=2,
                     label=label)
    rep += 1

data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.14",
])


data.pressure = np.round(data.pressure, 2)
#plot_pair(data, "pressure", ["C0", "C0"], label="5")
plot_stiffness(data, "pressure", ["C0", "C0"], "5", 0)

data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.30\200 pa",
])

data.pressure = np.round(data.pressure, 2)
#plot_pair(data, "pressure", ["C1", "C1"], label="200")
plot_stiffness(data, "pressure", ["C0", "C0"], "200", 1)

data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.30\700 pa",
])


data.pressure = np.round(data.pressure, 2)
#plot_pair(data, "pressure", ["C2", "C2"], label="700")
plot_stiffness(data, "pressure", ["C0", "C0"], "700", 2)
plt.legend()
plt.tight_layout()

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(11.990000/2.54, 5.980000/2.54, forward=True)
plt.figure(1).axes[0].set_xlim(0.0, 2.0)
plt.figure(1).axes[0].set_ylim(0.0, 0.8)
plt.figure(1).axes[0].set_xticklabels([])
plt.figure(1).axes[0].set_position([0.593028, 0.113252, 0.351161, 0.861581])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].title.set_visible(False)
plt.figure(1).axes[0].get_xaxis().get_label().set_text('')
plt.figure(1).axes[0].get_xaxis().get_label().set_visible(False)
plt.figure(1).axes[0].get_yaxis().get_label().set_text("alpha")
plt.figure(1).axes[1].set_xlim(0.0, 2.0)
plt.figure(1).axes[1].set_ylim(0.0, 14.0)
plt.figure(1).axes[1].set_xticklabels([])
plt.figure(1).axes[1].set_position([0.130955, 0.113252, 0.351161, 0.861581])
plt.figure(1).axes[1].spines['right'].set_visible(False)
plt.figure(1).axes[1].spines['top'].set_visible(False)
plt.figure(1).axes[1].title.set_visible(False)
plt.figure(1).axes[1].get_xaxis().get_label().set_text('')
plt.figure(1).axes[1].get_yaxis().get_label().set_text("stiffness")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()
