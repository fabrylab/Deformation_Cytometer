import glob
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


def stressfunc(R, P, L, H):  # imputs (radial position and pressure)
    G = P / L  # pressure gradient
    pre_factor = (4 * (H ** 2) * G) / np.pi ** 3
    # sum only over odd numbers
    n = np.arange(1, 100, 2)[None, :]
    u_primy = np.sum(pre_factor * ((-1) ** ((n - 1) / 2)) * (np.pi / ((n ** 2) * H)) \
                     * (np.sinh((n * np.pi * R[:, None]) / H) / np.cosh(n * np.pi / 2)), axis=1)

    stress = np.abs(u_primy)
    return stress


def getVelocity(data, config):
    # %% compute velocity profile
    y_pos = []
    vel = []
    velocities = np.zeros(data.shape[0]) * np.nan
    cell_id = data.index.to_numpy()
    velocity_partner = np.zeros(data.shape[0], dtype="<U100")
    for i in data.index[:-10]:
        for j in range(10):
            try:
                data.rp[i + j]
            except KeyError:
                continue
            if np.abs(data.rp[i] - data.rp[i + j]) < 1 \
                    and data.frames[i + j] - data.frames[i] == 1 \
                    and np.abs(data.long_axis[i + j] - data.long_axis[i]) < 1 \
                    and np.abs(data.short_axis[i + j] - data.short_axis[i]) < 1 \
                    and np.abs(data.angle[i + j] - data.angle[i]) < 5:

                dt = data.timestamp[i + j] - data.timestamp[i]
                v = (data.x[i + j] - data.x[i]) * config["pixel_size"] / dt  # in mm/s
                if v > 0:
                    velocities[i] = v
                    velocity_partner[
                        i] = f"{i}, {i + j}, {dt}, {data.x[i + j] - data.x[i]}, {data.frames[i]}, {data.long_axis[i]}, {data.short_axis[i]} -> {data.frames[i + j]}, {data.long_axis[i + j]}, {data.short_axis[i + j]}"
                    cell_id[i + j] = cell_id[i]
    data["velocity"] = velocities
    data["velocity_partner"] = velocity_partner
    data["cell_id"] = cell_id


def getStressStrain(data, config):
    data["stress"] = stressfunc(data.rp * 1e-6, -config["pressure_pa"], config["channel_length_m"],
                                config["channel_width_m"])
    data["strain"] = (data.long_axis - data.short_axis) / np.sqrt(data.long_axis * data.short_axis)


def filterCells(data, config):
    l_before = data.shape[0]
    data = data[(data.solidity > 0.96) & (data.irregularity < 1.06)]  # & (data.rp.abs() < 65)]
    # data = data[(data.solidity > 0.98) & (data.irregularity < 1.04) & (data.rp.abs() < 65)]

    l_after = data.shape[0]
    print('# frames =', data.frames.iloc[-1], '   # cells total =', l_before, '   #cells sorted = ', l_after)
    print('ratio #cells/#frames before sorting out = %.2f \n' % float(l_before / data.frames.iloc[-1]))

    config["filter"] = dict(l_before=l_before, l_after=l_after)

    data.reset_index(drop=True, inplace=True)

    return data


def fit_func_velocity(config):
    if "vel_fit" in config:
        p0, p1, p2 = config["vel_fit"]
        p2 = 0
    else:
        p0, p1, p2 = None, None, None

    def velfit(r, p0=p0, p1=p1, p2=p2):  # for stress versus strain
        R = config["channel_width_m"] / 2 * 1e6
        return p0 * (1 - np.abs((r + p2) / R) ** p1)

    return velfit


def fit_func_velocity_gradient(config):
    if "vel_fit" in config:
        p0, p1, p2 = config["vel_fit"]
        p2 = 0
    else:
        p0, p1, p2 = None, None, None

    def getVelGrad(r, p0=p0, p1=p1, p2=p2):
        r = r * 1e-6
        p0 = p0 * 1e-3
        r0 = 100e-6
        return - (p0 * p1 * (np.abs(r) / r0) ** p1) / r

    return getVelGrad


def correctCenter(data, config):
    if not "velocity" in data:
        getVelocity(data, config)
    d = data[~np.isnan(data.velocity)]
    y_pos = d.rp
    vel = d.velocity

    if len(vel) == 0:
        raise ValueError("No velocity values found.")

    vel_fit, pcov = curve_fit(fit_func_velocity(config), y_pos, vel,
                              [np.max(vel), 0.9, -np.mean(y_pos)])  # fit a parabolic velocity profile

    y_pos += vel_fit[2]
    # data.y += vel_fit[2]
    data.rp += vel_fit[2]

    config["vel_fit"] = list(vel_fit)
    config["center"] = vel_fit[2]

    data["velocity_gradient"] = fit_func_velocity_gradient(config)(data.rp)
    data["velocity_fitted"] = fit_func_velocity(config)(data.rp)
    data["imaging_pos_mm"] = config["imaging_pos_mm"]


def fit_func_strain(config):
    def fitfunc(x, p0, p1, p2):  # for stress versus strain
        k = p0  # (p0 + x)
        alpha = p1
        # return x / (k * (np.abs(w)) ** alpha)
        return x / (k * (np.abs(w) + (v / (np.pi * 2 * config["imaging_pos_mm"]))) ** alpha) + p2

    return fitfunc


def fitStiffness(data, config):
    def fitfunc(x, p0, p1):  # for stress versus strain
        return (1 / p0) * np.log((x / p1) + 1) + 0.05  # p2

    pstart = (0.1, 1)  # initial guess

    xy = np.vstack([data.stress, data.strain])
    kd = gaussian_kde(xy)(xy)

    # fit weighted by the density of points
    p, pcov = curve_fit(fitfunc, data.stress, data.strain, pstart, sigma=1 / kd, maxfev=10000)  # do the curve fitting
    # p, pcov = curve_fit(fitfunc, stress[RP<0], strain[RP<0], pstart) #do the curve fitting for one side only
    err = (np.diag(pcov)) ** 0.5  # estimate 1 standard error of the fit parameters
    cov_ap = pcov[0, 1]  # cov between alpha and prestress
    cov_ao = 0  # pcov[0, 2]*0  # cov between offset and alpha
    cov_po = 0  # pcov[1, 2]*0  # cov between prestress and offset
    se01 = np.sqrt((p[1] * err[0]) ** 2 + (p[0] * err[1]) ** 2 + 2 * p[0] * p[1] * cov_ap)
    print('pressure = %5.1f kPa' % float(config["pressure_pa"] / 1000))
    print("p0 =%5.2f   p1 =%5.1f Pa   p0*p1=%5.1f Pa   p2 =%4.3f" % (p[0], p[1], p[0] * p[1], 0))

    print("se0=%5.2f   se1=%5.1f Pa   se0*1=%5.1f Pa   se2=%4.3f" % (err[0], err[1], se01, 0))
    err = np.concatenate((err, [se01]))

    config["fit"] = dict(fitfunc=fitfunc, p=p, err=err, cov_ap=cov_ap, cov_ao=cov_ao, cov_po=cov_po)


def fitStiffness(data, config):
    def omega(x, p=52.43707149):
        return 0.25 * x / (2 * np.pi)

    v = data.velocity_fitted
    w = omega(data.velocity_gradient)
    imaging_pos_mm = data.imaging_pos_mm

    def fitfunc(x, p0, p1, p2):  # for stress versus strain
        k = p0  # (p0 + x)
        alpha = p1
        # return x / (k * (np.abs(w)) ** alpha)
        # return np.log(x/k + 1) / ((np.abs(w) + (v/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2
        return (x / (k * (np.abs(w) + (v / (np.pi * 2 * imaging_pos_mm))) ** alpha) + p2)  # * (x < 100)

    import scipy.optimize
    def curve_fit(func, x, y, start, maxfev=None, bounds=None):
        def cost(p):
            return np.mean(np.abs(func(x, *p) - y)) + (p[1] < 0) * 10

        res = scipy.optimize.minimize(cost, start)  # , bounds=np.array(bounds).T)  # , maxfev=maxfev, bounds=bounds)

        # print(res)
        return res["x"], []

    pstart = (np.random.uniform(50, 200), np.random.uniform(0.1, 0.5), 0)  # initial guess
    print("--", np.mean(data.stress), np.min(data.stress), np.max(data.stress))
    # fit weighted by the density of points
    p, pcov = curve_fit(fitfunc, data.stress, data.strain, pstart, bounds=[(0, 0, 0), (500, 1, 1)],
                        maxfev=10000)  # do the curve fitting
    # p, pcov = curve_fit(fitfunc, stress[RP<0], strain[RP<0], pstart) #do the curve fitting for one side only
    """
    err = (np.diag(pcov)) ** 0.5  # estimate 1 standard error of the fit parameters
    cov_ap = pcov[0, 1]  # cov between alpha and prestress
    cov_ao = 0#pcov[0, 2]*0  # cov between offset and alpha
    cov_po = 0#pcov[1, 2]*0  # cov between prestress and offset
    se01 = np.sqrt((p[1] * err[0]) ** 2 + (p[0] * err[1]) ** 2 + 2 * p[0] * p[1] * cov_ap)
    """
    print('pressure = %5.1f kPa' % float(config["pressure_pa"] / 1000))
    print("k = %6.2f   alpha = %4.2f Pa  offset = %4.3f" % (p[0], p[1], p[2]))

    # print("se0=%5.2f   se1=%5.1f Pa   se0*1=%5.1f Pa   se2=%4.3f" % (err[0], err[1], se01, 0))
    # err = np.concatenate((err, [se01]))
    cov_ap = 0
    cov_ao = 0
    cov_po = 0
    err = [0, 0, 0, 0]

    config["fit"] = dict(fitfunc=fitfunc, p=p, err=err, cov_ap=cov_ap, cov_ao=cov_ao, cov_po=cov_po)
    return p


def plotVelocityProfile(data, config):
    if "velocity" not in data:
        getVelocity(data, config)
    d = data[~np.isnan(data.velocity)]
    y_pos = d.rp
    vel = d.velocity

    def velfit(r, p0, p1):  # for stress versus strain
        R = config["channel_width_m"] / 2 * 1e6
        return p0 * (1 - np.abs(r / R) ** p1)

    vel_fit, pcov = curve_fit(velfit, y_pos, vel, [np.max(vel), 0.9])  # fit a parabolic velocity profile

    fig1 = plt.figure(1, (6, 4))
    border_width = 0.1
    ax_size = [0 + 2 * border_width, 0 + 2 * border_width,
               1 - 3 * border_width, 1 - 3 * border_width]
    ax1 = fig1.add_axes(ax_size)
    ax1.set_xlabel('channel position ($\u00B5 m$)')
    ax1.set_ylabel('flow speed (mm/s)')
    ax1.set_ylim(0, 1.1 * vel_fit[0])

    r = np.arange(-config["channel_width_m"] / 2 * 1e6, config["channel_width_m"] / 2 * 1e6,
                  0.1)  # generates an extended array

    ax1.plot(-y_pos, vel, '.', mfc="none", mec="C1", alpha=0.5)
    ax1.plot(y_pos, vel, '.', color="C0")
    ax1.plot(r, velfit(r, vel_fit[0], vel_fit[1]), '--', color='gray', linewidth=2, zorder=3)
    ax1.axvline(0, ls="--", color="k", lw=0.8)
    ax1.axhline(0, ls="--", color="k", lw=0.8)
    print('v_max = %5.2f mm/s   profile stretch exponent = %5.2f\n' % (vel_fit[0], vel_fit[1]))


def plotDensityScatter(x, y, cmap='viridis', alpha=1, skip=1):
    x = np.array(x)[::skip]
    y = np.array(y)[::skip]
    xy = np.vstack([x, y])
    kd = gaussian_kde(xy)(xy)
    idx = kd.argsort()
    x, y, z = x[idx], y[idx], kd[idx]
    plt.scatter(x, y, c=z, s=5, alpha=alpha, cmap=cmap)  # plot in kernel density colors e.g. viridis


def plotStressStrainFit(data, config):
    fit = config["fit"]
    fitfunc = fit["fitfunc"]
    p = fit["p"]
    err = fit["err"]
    cov_ap = fit["cov_ap"]
    cov_ao = fit["cov_ao"]
    cov_po = fit["cov_po"]

    # ----------plot the fit curve----------
    xx = np.arange(np.min(data.stress), np.max(data.stress), 0.1)  # generates an extended array
    plt.plot(xx, (fitfunc(xx, p[0], p[1])), '-', color='black', linewidth=2, zorder=3)

    # ----------plot standard error of the fit function----------
    dyda = -1 / (p[0] ** 2) * np.log(xx / p[1] + 1)  # strain derivative with respect to alpha
    dydp = -1 / p[0] * xx / (xx * p[1] + p[1] ** 2)  # strain derivative with respect to prestress
    dydo = 1  # strain derivative with respect to offset
    if 0:  # TODO
        vary = (dyda * err[0]) ** 2 + (dydp * err[1]) ** 2 + (
                dydo * err[2]) ** 2 + 2 * dyda * dydp * cov_ap + 2 * dyda * dydo * cov_ao + 2 * dydp * dydo * cov_po
        y1 = fitfunc(xx, p[0], p[1]) - np.sqrt(vary)
        y2 = fitfunc(xx, p[0], p[1]) + np.sqrt(vary)
        plt.fill_between(xx, y1, y2, facecolor='gray', edgecolor="none", linewidth=0, alpha=0.5)


def plotStressStrainFit(data, config, color="C1"):
    def omega(x, p=52.43707149):
        return 0.25 * x / (2 * np.pi)

    v = data.velocity_fitted
    w = omega(data.velocity_gradient)
    imaging_pos_mm = data.imaging_pos_mm

    def fitfunc(x, p0, p1, p2):  # for stress versus strain
        k = p0  # (p0 + x)
        alpha = p1
        # return x / (k * (np.abs(w)) ** alpha)
        # return np.log(x/k + 1) / ((np.abs(w) + (v/(np.pi*2*config1["imaging_pos_mm"]))) ** alpha) + p2
        return (x / (k * (np.abs(w) + (v / (np.pi * 2 * imaging_pos_mm))) ** alpha) + p2)  # * (x < 100)

    x = data.stress

    p0, p1, p2 = config["fit"]["p"]
    y = fitfunc(x, p0, p1, p2)
    indices = np.argsort(x)
    x = x[indices]
    y = y[indices]
    x2 = []
    y2 = []
    delta = 10
    for i in np.arange(x.min(), x.max() - delta, delta):
        indices = (i < x) & (x < (i + delta))
        if len(indices) >= 10:
            x2.append(np.mean(x[indices]))
            y2.append(np.mean(y[indices]))
    plt.plot(x2, y2, "-", color=color, lw=4)


def bootstrap_median_error(data):
    data = np.asarray(data)
    if len(data) <= 1:
        return 0
    medians = []
    for i in range(1000):
        medians.append(np.median(data[np.random.random_integers(len(data) - 1, size=len(data))]))
    return np.nanstd(medians)


def plotBinnedData(x, y, bins):
    strain_av = []
    stress_av = []
    strain_err = []
    for i in range(len(bins) - 1):
        index = (x > bins[i]) & (x < bins[i + 1])
        yy = y[index]
        strain_av.append(np.median(yy))
        # yy = yy[yy>0]
        # strain_err.append(np.std(np.log(yy)) / np.sqrt(len(yy)))
        strain_err.append(bootstrap_median_error(yy))  # np.quantile(yy, [0.25, 0.75]))

        stress_av.append(np.median(x[index]))
    plt.errorbar(stress_av, strain_av, yerr=strain_err, marker='s', mfc='white', \
                 mec='black', ms=7, mew=1, lw=0, ecolor='black', elinewidth=1, capsize=3)


def plotStressStrain(data, config, skip=1):
    # %% fitting deformation with stress stiffening equation
    # fig2 = plt.figure(2, (6, 6))
    border_width = 0.1
    ax_size = [0 + 2 * border_width, 0 + 2 * border_width,
               1 - 3 * border_width, 1 - 3 * border_width]
    # ax2 = fig2.add_axes(ax_size)
    ax2 = plt.gca()
    ax2.set_xlabel('fluid shear stress $\u03C3$ (Pa)')
    ax2.set_ylabel('cell strain  $\u03B5$')

    pmax = 50 * np.ceil((np.max(data.stress) + 50) // 50)
    # ax2.set_xticks(np.arange(0, pmax + 1, 50))
    # ax2.set_xlim((-10, pmax))
    # ax2.set_ylim((-0.2, 1.0))

    # ----------plot strain versus stress data points----------
    plotDensityScatter(data.stress, data.strain, skip=skip)

    # ----------plot the fit curve----------
    plotStressStrainFit(data, config)

    # ----------plot the binned (averaged) strain versus stress data points----------
    plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])


def plotMessurementStatus(data, config):
    firstPage = plt.figure(figsize=(11.69, 8.27))
    firstPage.clf()
    txt = []
    if "filter" in config:
        txt.append(
            f'# frames = {data.frames.iloc[-1]}   # cells total = {config["filter"]["l_before"]}   #cells sorted = {config["filter"]["l_after"]}')
        txt.append('ratio #cells/#frames before sorting out = %.2f \n' % float(
            config["filter"]["l_before"] / data.frames.iloc[-1]))
    txt.append('center channel position at y = %.1f  \u03BCm' % -config["center"])
    txt.append('v_max = %5.2f mm/s   profile stretch exponent = %5.2f\n' % (config["vel_fit"][0], config["vel_fit"][1]))
    txt.append('pressure = %5.1f kPa' % float(config["pressure_pa"] / 1000))

    fit = config["fit"]
    p = fit["p"]
    err = fit["err"]
    if len(p) == 3:
        txt.append(f"k = {p[0]:3.0f} Pa   $\\alpha$ = {p[1]:3.2f}  offset = {p[2]:4.3f}")
        # txt.append("p0 =%5.2f   p1 =%5.1f Pa   p0*p1=%5.1f Pa   p2 =%4.3f" % (p[0], p[1], p[0] * p[1], p[2]))
        # txt.append("se0=%5.2f   se1=%5.1f Pa   se0*1=%5.1f Pa   se2=%4.3f" % (err[0], err[1], err[0] * err[1], err[2]))
    else:
        txt.append("p0 =%5.2f   p1 =%5.1f Pa   p0*p1=%5.1f Pa" % (p[0], p[1], p[0] * p[1]))
        txt.append("se0=%5.2f   se1=%5.1f Pa   se0*1=%5.1f Pa" % (err[0], err[1], err[0] * err[1]))
    txt_whole = "\n".join(txt)
    firstPage.text(0.5, 0.5, txt_whole, transform=firstPage.transFigure, size=24, ha="center")


def initPlotSettings():
    # ----------general fonts for plots and figures----------
    font = {'family': 'sans-serif',
            'sans-serif': ['Arial'],
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=12)
    plt.rc('axes', titlesize=18)
    C0 = '#1f77b4'
    C1 = '#ff7f0e'
    C2 = '#2ca02c'
    C3 = '#d62728'


def storeEvaluationResults(data, config):
    # %% store the results
    output_path = os.getcwd()
    date_time = str(Path(config["file_data"]).name).split('_')
    print(config["file_data"])
    seconds = float(date_time[3]) * 60 * 60 + float(date_time[4]) * 60 + float(date_time[5])
    alldata_file = output_path + '/' + 'all_data.txt'
    if not os.path.exists(alldata_file):
        f = open(alldata_file, 'at')
        f.write(
            'filename' + '\t' + 'seconds' + '\t' + 'p (kPa)' + '\t' + '#cells' + '\t' + '#diameter (um)' + '\t' + 'vmax (mm/s)' + '\t' + 'expo' + '\t' + 'alpha' + '\t' + 'sigma' + '\t' + 'eta_0' + '\t' + 'stiffness (Pa)' + '\n')
    else:
        f = open(alldata_file, 'at')
    f.write(Path(config["file_data"]).name + '\t' + '{:.0f}'.format(seconds) + '\t')

    D = np.sqrt(data.long_axis * data.short_axis)
    f.write(str(config["pressure_pa"] / 1000) + '\t' + str(len(data.rp)) + '\t' + '{:0.1f}'.format(np.mean(D)) + '\t')
    f.write('{:0.3f}'.format(config["vel_fit"][0]) + '\t' + '{:0.3f}'.format(config["vel_fit"][1]) + '\t')
    if len(config["fit"]["p"]) == 3:
        f.write('{:0.3f}'.format(config["fit"]["p"][0]) + '\t' + '{:0.2f}'.format(
            config["fit"]["p"][1]) + '\t' + '{:0.3f}'.format(
            config["fit"]["p"][2]) + '\t' + '{:0.3f}'.format(config["fit"]["p"][0] * config["fit"]["p"][1]) + '\n')
    else:
        f.write('{:0.3f}'.format(config["fit"]["p"][0]) + '\t' + '{:0.2f}'.format(
            config["fit"]["p"][1]) + '\t' + '\t' + '{:0.3f}'.format(
            config["fit"]["p"][0] * config["fit"]["p"][1]) + '\n')
    # f.write(str(frame[i]) +'\t' +str(X[i]) +'\t' +str(Y[i]) +'\t' +str(R[i]) +'\t' +str(LongAxis[i]) +'\t'+str(ShortAxis[i]) +'\t' +str(Angle[i]) +'\t' +str(irregularity[i]) +'\t' +str(solidity[i]) +'\t' +str(sharpness[i]) +'\n')
    f.close()


def get_pressures(input_path, repetition=None):
    paths = get_folders(input_path, repetition=None)

    pressures = []
    for index, file in enumerate(paths):
        config = getConfig(file)
        pressures.append(config['pressure_pa'] / 100_000)

    return np.array(pressures)


def get_folders(input_path, pressure=None, repetition=None):
    if isinstance(input_path, str):
        input_path = [input_path]
    paths = []
    for path in input_path:
        if "*" in path:
            glob_data = glob.glob(path, recursive=True)
            # print("glob_data", glob_data, path)
            if repetition is not None:
                glob_data = glob_data[repetition:repetition + 1]
            paths.extend(glob_data)
            # print("glob_data", glob_data)
        else:
            paths.append(path)

    new_paths = []
    for file in paths:
        # print("->", file)
        try:
            config = getConfig(file)
            new_paths.append(file)
        except OSError as err:
            print(err, file=sys.stderr)
            continue
    paths = new_paths
    # print("new paths", new_paths)

    if pressure is not None:
        pressures = []
        for index, file in enumerate(paths):
            config = getConfig(file)
            pressures.append(config['pressure_pa'] / 100_000)
            # print("->", file, pressures[-1])

        paths = np.array(paths)
        pressures = np.array(pressures)
        # print("pressures", pressures, pressure)
        paths = paths[pressures == pressure]

    return paths


def load_all_data(input_path, pressure=None, repetition=None):
    global ax

    evaluation_version = 2

    paths = get_folders(input_path, pressure=pressure, repetition=repetition)

    # print(paths)
    fit_data = []

    data_list = []
    for index, file in enumerate(paths):
        # print(file)
        output_file = Path(str(file).replace("_result.txt", "_evaluated.csv"))
        output_config_file = Path(str(file).replace("_result.txt", "_evaluated_config.txt"))

        # load the data and the config
        data = getData(file)
        config = getConfig(file)

        version = 0
        if output_config_file.exists():
            with output_config_file.open("r") as fp:
                config = json.load(fp)
            if "evaluation_version" in config:
                version = config["evaluation_version"]

        """ evaluating data"""
        if not output_file.exists() or version < evaluation_version:
            # refetchTimestamps(data, config)

            getVelocity(data, config)

            # take the mean of all values of each cell
            data = data.groupby(['cell_id']).mean()

            correctCenter(data, config)

            data = filterCells(data, config)

            # reset the indices
            data.reset_index(drop=True, inplace=True)

            getStressStrain(data, config)

            # data = data[(data.stress < 50)]
            data.reset_index(drop=True, inplace=True)

            data["area"] = data.long_axis * data.short_axis * np.pi
            try:
                config["evaluation_version"] = evaluation_version
                data.to_csv(output_file, index=False)
                # print("config", config, type(config))
                with output_config_file.open("w") as fp:
                    json.dump(config, fp)

            except PermissionError:
                pass
        else:
            with output_config_file.open("r") as fp:
                config = json.load(fp)

        data = pd.read_csv(output_file)

        # data = data[(data.area > 0) * (data.area < 2000) * (data.stress < 250)]
        # data.reset_index(drop=True, inplace=True)

        data_list.append(data)

    data = pd.concat(data_list)
    data.reset_index(drop=True, inplace=True)

    # fitStiffness(data, config)
    return data, config


def all_plots_same_limits():
    xmin = np.min([ax.get_xlim()[0] for ax in plt.gcf().axes])
    xmax = np.max([ax.get_xlim()[1] for ax in plt.gcf().axes])
    ymin = np.min([ax.get_ylim()[0] for ax in plt.gcf().axes])
    ymax = np.max([ax.get_ylim()[1] for ax in plt.gcf().axes])
    for ax in plt.gcf().axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


def do_bootstrap_fit(data, config):
    d0 = data.sample(frac=1, replace=True)
    return fitStiffness(d0, config)


def get_bootstrap_fit(data, config, N):
    res = []
    for i in range(N):
        res.append(do_bootstrap_fit(data, config))
    return np.array(res)
