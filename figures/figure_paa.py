import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new, bootstrap_match_hist, bootstrap_error
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
        agg_count = data.groupby(parameter)[value].count()  # agg(lambda x: 0.5*np.sqrt((x**2).sum()))
        l = plt.errorbar(agg_mean.index, agg_mean.values, agg_err_mean.values, capsize=3, color=color, zorder=2,
                         label=label)
        for (i, m, c) in zip(agg_mean.index, agg_mean.values, agg_count.values):
            plt.text(i, m, c)
    else:
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


def plot_pair(data, parameter, color=["C0", "C1"], label=None):
    plt.axes([0.1, .2, 0.4, 0.5], label=f"{parameter}_alpha")
    plotMeasurementOfParameter(data, parameter, "w_k_cell", agg="median", color=color[0], label=label)
    plt.ylim(bottom=0)
    plt.axes([0.5, .2, 0.5, 0.5], label=f"{parameter}_k")
    plotMeasurementOfParameter(data, parameter, "w_alpha_cell", agg="mean", color=color[1], label=label)
    plt.ylim(bottom=0)


""" """

data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.14\**\*evaluated_new_hand2.csv"
#r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.14"
])




data.pressure = np.round(data.pressure, 2)
plot_pair(data, "pressure", ["C4", "C4"], label="manual")
data["r0"] = np.sqrt(data.long_axis * data.short_axis)
print(data.groupby("pressure").area.mean())
print(data.groupby("pressure").area.std())

print(data.groupby("pressure").r0.mean())
print(data.groupby("pressure").r0.std())


for name, d in data.groupby("pressure"):
    k, alpha = getFitParameters(d)
    data.at[data.pressure == name, "w_k_cell"] = k
    data.at[data.pressure == name, "w_alpha_cell"] = alpha
#plot_pair(data, "pressure", ["C3", "C3"])
#plt.show()

"""x"""

data, config = load_all_data_new([
r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\flourescence beads\2021.4.14\**\*evaluated_new.csv"
])




data.pressure = np.round(data.pressure, 2)
plot_pair(data, "pressure", ["C0", "C0"], "automatic")
data["r0"] = np.sqrt(data.long_axis * data.short_axis)
print(data.groupby("pressure").area.mean())
print(data.groupby("pressure").area.std())

print(data.groupby("pressure").r0.mean())
print(data.groupby("pressure").r0.std())


for name, d in data.groupby("pressure"):
    k, alpha = getFitParameters(d)
    data.at[data.pressure == name, "w_k_cell"] = k
    data.at[data.pressure == name, "w_alpha_cell"] = alpha
#plot_pair(data, "pressure", ["C3", "C3"])

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.000000/2.54, 5.300000/2.54, forward=True)
plt.figure(1).ax_dict["pressure_alpha"].set_xlim(0.0, 0.52)
plt.figure(1).ax_dict["pressure_alpha"].set_ylim(0.0, 12.0)
plt.figure(1).ax_dict["pressure_alpha"].set_position([0.125368, 0.310687, 0.364634, 0.554430])
plt.figure(1).ax_dict["pressure_alpha"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pressure_alpha"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[0].set_position([0.098868, 4.896183])
plt.figure(1).ax_dict["pressure_alpha"].texts[1].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[1].set_position([0.200000, 5.998325])
plt.figure(1).ax_dict["pressure_alpha"].texts[2].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[2].set_position([0.296605, 2.439314])
plt.figure(1).ax_dict["pressure_alpha"].texts[3].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[3].set_position([0.404527, 4.896183])
plt.figure(1).ax_dict["pressure_alpha"].texts[4].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[4].set_position([0.500000, 5.364158])
plt.figure(1).ax_dict["pressure_alpha"].texts[5].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[5].set_position([0.098868, 10.472887])
plt.figure(1).ax_dict["pressure_alpha"].texts[6].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[6].set_position([0.200000, 10.472887])
plt.figure(1).ax_dict["pressure_alpha"].texts[7].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[7].set_position([0.296605, 9.687726])
plt.figure(1).ax_dict["pressure_alpha"].texts[8].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[8].set_position([0.404527, 11.871610])
plt.figure(1).ax_dict["pressure_alpha"].texts[9].set_ha("center")
plt.figure(1).ax_dict["pressure_alpha"].texts[9].set_position([0.500000, 9.687726])
plt.figure(1).ax_dict["pressure_alpha"].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).ax_dict["pressure_alpha"].get_yaxis().get_label().set_text("stiffness (Pa)")
plt.figure(1).ax_dict["pressure_k"].set_xlim(0.0, 0.52)
plt.figure(1).ax_dict["pressure_k"].set_ylim(0.0, 0.6)
plt.figure(1).ax_dict["pressure_k"].legend()
plt.figure(1).ax_dict["pressure_k"].set_position([0.614226, 0.310687, 0.364634, 0.554430])
plt.figure(1).ax_dict["pressure_k"].spines['right'].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].spines['top'].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].yaxis.labelpad = -0.763429
plt.figure(1).ax_dict["pressure_k"].texts[0].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[1].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[2].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[3].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[4].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[5].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[6].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[7].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[8].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].texts[9].set_visible(False)
plt.figure(1).ax_dict["pressure_k"].get_xaxis().get_label().set_text("pressure (Pa)")
plt.figure(1).ax_dict["pressure_k"].get_yaxis().get_label().set_text("alpha")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()
