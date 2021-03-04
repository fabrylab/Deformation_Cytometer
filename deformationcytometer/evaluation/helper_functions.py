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
from deformationcytometer.includes.includes import getData, getConfig



rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']


def stressfunc(R, P, L, H):  # imputs (radial position and pressure)
    R = np.asarray(R)
    G = P / L  # pressure gradient
    pre_factor = (4 * (H ** 2) * G) / np.pi ** 3
    # sum only over odd numbers
    n = np.arange(1, 100, 2)[None, :]
    u_primy = pre_factor * np.sum(((-1) ** ((n - 1) / 2)) * (np.pi / ((n ** 2) * H)) \
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
    r = np.sqrt(data.long_axis/2 * data.short_axis/2) * 1e-6
    data["stress"] = 0.5*stressfunc(data.rp * 1e-6 + r, -config["pressure_pa"], config["channel_length_m"], config["channel_width_m"])\
                     + 0.5*stressfunc(data.rp * 1e-6 - r, -config["pressure_pa"], config["channel_length_m"],
                                config["channel_width_m"])
    data["stress_center"] = stressfunc(data.rp * 1e-6, -config["pressure_pa"], config["channel_length_m"],
                                      config["channel_width_m"])

    data["strain"] = (data.long_axis - data.short_axis) / np.sqrt(data.long_axis * data.short_axis)


def filterCells(data, config=None, solidity_threshold=0.96, irregularity_threshold=1.06):
    l_before = data.shape[0]
    data = data[(data.solidity > solidity_threshold) & (data.irregularity < irregularity_threshold)]  # & (data.rp.abs() < 65)]
    # data = data[(data.solidity > 0.98) & (data.irregularity < 1.04) & (data.rp.abs() < 65)]

    l_after = data.shape[0]

    if config is not None:
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
        p0 = p0 * 1e3
        R = config["channel_width_m"] / 2 * 1e6
        return - (p0 * p1 * (np.abs(r) / R) ** p1) / r

    return getVelGrad


def correctCenter(data, config):
    if not "velocity" in data:
        getVelocity(data, config)
    d = data[np.isfinite(data.velocity)]
    y_pos = d.rp
    vel = d.velocity

    if len(vel) == 0:
        raise ValueError("No velocity values found.")

    vel_fit, pcov = curve_fit(fit_func_velocity(config), y_pos, vel,
                              [np.nanpercentile(vel, 95), 3, -np.mean(y_pos)])  # fit a parabolic velocity profile
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
    gamma_dot = data.velocity_gradient
    b = data.short_axis/np.sqrt(data.long_axis*data.short_axis)
    v = data.velocity_fitted
    x_pos = data.imaging_pos_mm

    f = 1/(np.pi*2) * v / x_pos + 1/(np.pi*4) * b ** 2.56 * np.abs(gamma_dot)
    #f = 1/(np.pi*2) * v / x_pos + 0.04 * np.abs(gamma_dot)

    def fitfunc(x, p0, p1, p2):
        k = p0
        alpha = p1
        return x / (k * f ** alpha) + p2

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
        print("Radius ", R)
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


def plotDensityScatter(x, y, cmap='viridis', alpha=1, skip=1, y_factor=1, s=5, levels=None, loglog=False, ax=None):
    ax = plt.gca() if ax is None else ax
    x = np.array(x)[::skip]
    y = np.array(y)[::skip]
    filter = ~np.isnan(x) & ~np.isnan(y)
    if loglog is True:
        filter &= (x>0) & (y>0)
    x = x[filter]
    y = y[filter]
    if loglog is True:
        xy = np.vstack([np.log10(x), np.log10(y)])
    else:
        xy = np.vstack([x, y*y_factor])
    kde = gaussian_kde(xy)
    kd = kde(xy)
    idx = kd.argsort()
    x, y, z = x[idx], y[idx], kd[idx]
    ax.scatter(x, y, c=z, s=s, alpha=alpha, cmap=cmap)  # plot in kernel density colors e.g. viridis

    if levels != None:
        X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
        print(xy.shape)
        print(np.dstack([X, Y*y_factor]).shape)
        XY = np.dstack([X, Y*y_factor])
        Z = kde(XY.reshape(-1, 2).T).reshape(XY.shape[:2])
        ax.contour(X, Y, Z, levels=1)

    if loglog is True:
        ax.loglog()
    return ax

def plotDensityLevels(x, y, skip=1, y_factor=1, levels=None, cmap="viridis", colors=None):
    x = np.array(x)[::skip]
    y = np.array(y)[::skip]
    filter = ~np.isnan(x) & ~np.isnan(y)
    x = x[filter]
    y = y[filter]
    xy = np.vstack([x, y*y_factor])
    kde = gaussian_kde(xy)
    kd = kde(xy)
    idx = kd.argsort()
    x, y, z = x[idx], y[idx], kd[idx]

    X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    print(xy.shape)
    print(np.dstack([X, Y*y_factor]).shape)
    XY = np.dstack([X, Y*y_factor])
    Z = kde(XY.reshape(-1, 2).T).reshape(XY.shape[:2])
    plt.contour(X, Y, Z, levels=levels, cmap=cmap, colors=colors)

    #

def plotStressStrainFit(data, config, color="C1"):
    gamma_dot = data.velocity_gradient
    b = data.short_axis / np.sqrt(data.long_axis * data.short_axis)
    v = data.velocity_fitted
    x_pos = data.imaging_pos_mm

    f = 1 / (np.pi * 2) * v / x_pos + 1 / (np.pi * 4) * b ** 2.56 * np.abs(gamma_dot)
    #f = 1 / (np.pi * 2) * v / x_pos + 0.04 * np.abs(gamma_dot)

    def fitfunc(x, p0, p1, p2):
        k = p0
        alpha = p1
        return x / (k * f ** alpha) + p2

    x = data.stress
    # calculating the strain with fitted parameters
    p0, p1, p2 = config["fit"]["p"]
    y = fitfunc(x, p0, p1, p2)
    # sorting the values from lower to higher stress
    indices = np.argsort(x)
    if 1:
        x = x[indices]
        y = y[indices]
        x2 = []
        y2 = []
        # binning every 10 th value
        delta = 10
        for i in np.arange(x.min(), x.max() - delta, delta):
            indices = (i < x) & (x < (i + delta))
            if len(indices) >= 10:
                x2.append(np.mean(x[indices]))
                y2.append(np.mean(y[indices]))
        #plt.plot(x2, y2, "-", color=color, lw=4)
    plt.plot(x, y, ".", color=color, lw=4)


def bootstrap_error(data, func=np.median, repetitions=1000):
    data = np.asarray(data)
    if len(data) <= 1:
        return 0
    medians = []
    for i in range(repetitions):
        medians.append(func(data[np.random.random_integers(len(data) - 1, size=len(data))]))
    return np.nanstd(medians)


def plotBinnedData(x, y, bins, bin_func=np.median, error_func=None, color="black", **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)
    strain_av = []
    stress_av = []
    strain_err = []
    for i in range(len(bins) - 1):
        index = (bins[i] < x) & (x < bins[i + 1])
        yy = y[index]
        yy = yy[~np.isnan(yy)]
        if len(yy) == 0:
            continue
        strain_av.append(bin_func(yy))
        # yy = yy[yy>0]
        # strain_err.append(np.std(np.log(yy)) / np.sqrt(len(yy)))
        if error_func is None:
            strain_err.append(bootstrap_error(yy, bin_func))  # np.quantile(yy, [0.25, 0.75]))
        elif error_func == "quantiles":
            strain_err.append(np.abs(np.quantile(yy, [0.25, 0.75])-bin_func(yy)))  # np.quantile(yy, [0.25, 0.75]))

        stress_av.append(np.median(x[index]))
    plot_kwargs = dict(marker='s', mfc="white", mec=color, ms=7, mew=1, lw=0, ecolor='black', elinewidth=1, capsize=3)
    plot_kwargs.update(kwargs)
    plt.errorbar(stress_av, strain_av, yerr=np.array(strain_err).T, **plot_kwargs)
    x, y = np.array(stress_av), np.array(strain_av)
    index = ~np.isnan(x) & ~np.isnan(y)
    x = x[index]
    y = y[index]
    return x, y


def plotStressStrain(data, config, skip=1, color="C1", mew=1):
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
    plotStressStrainFit(data, config, color=color)

    # ----------plot the binned (averaged) strain versus stress data points----------
    plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250], color=color, mew=mew)


def plotMessurementStatus(data, config):
    firstPage = plt.figure(figsize=(11.69, 8.27))
    firstPage.clf()
    txt = []
    if "filter" in config:
        txt.append(
            f'# frames in one experiment = {int(data.frames.iloc[-1])}   # cells total = {config["filter"]["l_before"]}\n   #cells sorted = {config["filter"]["l_after"]}')
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
    # first splitting step in case Path fails to find filename --> didn't work on linux when accesing the server
    date_time = str(Path(config["file_data"]).name).split('\\')
    date_time = date_time[-1].split('_')
    seconds = float(date_time[3]) * 60 * 60 + float(date_time[4]) * 60 + float(date_time[5])
    alldata_file = output_path + '/' + 'all_data.txt'
    if not os.path.exists(alldata_file):
        f = open(alldata_file, 'at')
        f.write(
            'filename' + '\t' + 'seconds' + '\t' + 'p (kPa)' + '\t' + '#cells' + '\t' + '#diameter (um)' + '\t' + 'vmax (mm/s)' + '\t' + 'expo' + '\t' + '(stiffness) k' + '\t' + 'alpha' + '\t' + 'offset (epsilon0)' + '\t' + 'alpha * k' + '\n')
    else:
        f = open(alldata_file, 'at')
    f.write(Path(config["file_data"]).name + '\t' + '{:.0f}'.format(seconds) + '\t')

    D = np.sqrt(data.long_axis * data.short_axis)
    f.write(str(config["pressure_pa"] / 1000) + '\t' + str(len(data.rp)) + '\t' + '{:0.1f}'.format(np.mean(D)) + '\t')
    f.write('{:0.3f}'.format(config["vel_fit"][0]) + '\t' + '{:0.3f}'.format(config["vel_fit"][1]) + '\t')
    if len(config["fit"]["p"]) == 3:
        f.write('{:0.3f}'.format(config["fit"]["p"][0])
                + '\t' + '{:0.2f}'.format(config["fit"]["p"][1]) +
                '\t' + '{:0.3f}'.format(config["fit"]["p"][2]) +
                '\t' + '{:0.3f}'.format(config["fit"]["p"][0] * config["fit"]["p"][1]) + '\n') # TODO: what is dis?
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


def check_config_changes(config, evaluation_version, solidity_threshold, irregularity_threshold):
    # checking if evaluation version, solidity or regularity thresholds have changed or if flag
    # for new network evaluation is set to True

    checks = {"evaluation_version": evaluation_version, "solidity": solidity_threshold,
              "irregularity": irregularity_threshold, "network_evaluation_done": True}

    for key, value in checks.items():
        if key not in config:
            return True
        else:
            if config[key] != value:
                return True
    return False


def load_all_data(input_path, solidity_threshold=0.96, irregularity_threshold=1.06, pressure=None, repetition=None):
    global ax

    evaluation_version = 8

    paths = get_folders(input_path, pressure=pressure, repetition=repetition)
    fit_data = []
    data_list = []
    filters = []
    config = {}
    for index, file in enumerate(paths):
        #print(file)
        output_file = Path(str(file).replace("_result.txt", "_evaluated.csv"))
        output_config_file = Path(str(file).replace("_result.txt", "_evaluated_config.txt"))

        # load the data and the config
        data = getData(file)
        config = getConfig(file)
        config["channel_width_m"] = 0.00019001261833616293

        if output_config_file.exists():
            with output_config_file.open("r") as fp:
                config = json.load(fp)
                config["channel_width_m"] = 0.00019001261833616293

        config_changes = check_config_changes(config, evaluation_version, solidity_threshold, irregularity_threshold)
        if "filter" in config:
                filters.append(config["filter"])


        """ evaluating data"""
        if not output_file.exists() or config_changes:
            #refetchTimestamps(data, config)
            #data = data[data.frames % 2 == 0]
            #data.frames = data.frames // 2
            #data.reset_index(drop=True, inplace=True)

            getVelocity(data, config)
            # take the mean of all values of each cell
            data = data.groupby(['cell_id'], as_index=False).mean()

            tt_file = Path(str(file).replace("_result.txt", "_tt.csv"))
            if tt_file.exists():
                data.set_index("cell_id", inplace=True)
                data_tt = pd.read_csv(tt_file)
                data["omega"] = np.zeros(len(data)) * np.nan
                for i, d in data_tt.iterrows():
                    if d.tt_r2 > 0.2:
                        data.at[d.id, "omega"] = d.tt * 2 * np.pi

                data.reset_index(inplace=True)
            else:
                print("WARNING: tank treading has not been evaluated yet")

            correctCenter(data, config)

            data = filterCells(data, config, solidity_threshold, irregularity_threshold)
            # reset the indices
            data.reset_index(drop=True, inplace=True)

            getStressStrain(data, config)

            #data = data[(data.stress < 50)]
            data.reset_index(drop=True, inplace=True)

            data["area"] = data.long_axis * data.short_axis * np.pi
            data["pressure"] = config["pressure_pa"]*1e-5

            data, p = apply_velocity_fit(data)


            omega, mu1, eta1, k_cell, alpha_cell, epsilon = get_cell_properties(data)

            try:
                config["evaluation_version"] = evaluation_version
                config["network_evaluation_done"] = True
                config["solidity"] = solidity_threshold
                config["irregularity"] = irregularity_threshold
                data.to_csv(output_file, index=False)
                #print("config", config, type(config))
                with output_config_file.open("w") as fp:
                    json.dump(config, fp, indent=0)

            except PermissionError:
                pass

        else:
            with output_config_file.open("r") as fp:
                config = json.load(fp)
                config["channel_width_m"] = 0.00019001261833616293

        data = pd.read_csv(output_file)

        #data = data[(data.area > 0) * (data.area < 2000) * (data.stress < 250)]
        #data.reset_index(drop=True, inplace=True)

        data_list.append(data)
    l_before = np.sum([d["l_before"] for d in filters])
    l_after = np.sum([d["l_after"] for d in filters])

    config["filter"] = {"l_before":l_before, "l_after":l_after}
    try:
        data = pd.concat(data_list)
    except ValueError:
        raise ValueError("No object found", input_path)
    data.reset_index(drop=True, inplace=True)

    #fitStiffness(data, config)
    return data, config


def load_all_data_new(input_path, solidity_threshold=0.96, irregularity_threshold=1.06, pressure=None, repetition=None, do_group=True):

    paths = get_folders(input_path, pressure=pressure, repetition=repetition)
    data_list = []
    config = {}
    for index, file in enumerate(paths):
        output_file = Path(str(file).replace("_result.txt", "_evaluated_new.csv"))
        output_config_file = Path(str(output_file).replace("_evaluated_new.csv", "_evaluated_config_new.txt"))
        print(output_file)

        with output_config_file.open("r") as fp:
            config = json.load(fp)
            config["channel_width_m"] = 0.00019001261833616293

        data = pd.read_csv(output_file)
        if do_group is True:
            data = data.groupby(['cell_id'], as_index=False).mean()

        data_list.append(data)

    data = pd.concat(data_list)
    data.reset_index(drop=True, inplace=True)

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


from deformationcytometer.includes.fit_velocity import fit_velocity, fit_velocity_pressures, getFitXY
def plot_velocity_fit(data, p, color=None):
    def getFitLine(pressure, p):
        config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
        x, y = getFitXY(config, np.mean(pressure), p)
        return x, y

    for pressure in data.pressure.unique():
        d = data[data.pressure == pressure]
        x, y = getFitLine(pressure, p)
        plt.plot(d.rp , d.velocity * 1e-3, "o", alpha=0.3, ms=2, color=color)
        l, = plt.plot(x * 1e+6, y, color="k")
    plt.xlabel("position in channel (µm)")
    plt.ylabel("velocity (m/s)")

def plot_velocity_fit(data, color=None):
    def getFitLine(pressure, p):
        config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
        x, y = getFitXY(config, np.mean(pressure), p)
        return x, y

    maxima = []
    for pressure in sorted(data.pressure.unique(), reverse=True):
        d = data[data.pressure == pressure]
        d.delta = d.delta.round(8)
        d.tau = d.tau.round(10)
        d.eta0 = d.eta0.round(8)
        d = d.set_index(["eta0", "delta", "tau"])
        for p in d.index.unique():
            dd = d.loc[p]
            x, y = getFitLine(pressure, p)
            line, = plt.plot(np.abs(dd.rp), dd.velocity * 1e-3 * 1e2, "o", alpha=0.3, ms=2, color=color)
            plt.plot([], [], "o", ms=2, color=line.get_color(), label=f"{pressure:.1f}")
            l, = plt.plot(x[x>=0]* 1e+6, y[x>=0] * 1e2, color="k")
            maxima.append(np.max(y[x>0]* 1e2))
    plt.ylim(top=np.max(maxima)*1.1)
    plt.xlabel("position in channel (µm)")
    plt.ylabel("velocity (cm/s)")


def apply_velocity_fit(data2):
    config = {"channel_length_m": 5.8e-2, "channel_width_m": 186e-6}
    p0, vel, vel_grad = fit_velocity_pressures(data2, config, x_sample=100)
    eta0, delta, tau = p0
    eta = eta0 / (1 + tau ** delta * np.abs(vel_grad) ** delta)
    data2["vel"] = vel
    data2["vel_grad"] = vel_grad
    data2["eta"] = eta
    data2["eta0"] = eta0
    data2["delta"] = delta
    data2["tau"] = tau
    return data2, p0

def plot_density_hist(x, orientation='vertical', do_stats=True, only_kde=False, ax=None, **kwargs):
    ax = ax if not ax is None else plt.gca()
    from scipy import stats
    x = np.array(x)
    x = x[~np.isnan(x)]
    kde = stats.gaussian_kde(x)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    if orientation == 'horizontal':
        l, = ax.plot(kde(xx), xx, **kwargs)
    else:
        l, = ax.plot(xx, kde(xx), **kwargs)
    if not only_kde:
        ax.hist(x, bins=50, density=True, color=l.get_color(), alpha=0.5, orientation=orientation)
    return l

def plot_joint_density(x, y, label=None, only_kde=False, color=None, growx=1, growy=1, offsetx=0):
    ax = plt.gca()
    x1, y1, w, h = ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height

    wf, hf = ax.figure.get_size_inches()
    gap = 0.05
    fraction = 0.2
    width_of_hist = np.mean([(w*wf*fraction), (h*hf*fraction)])
    hist_w = width_of_hist/wf
    hist_h = width_of_hist/hf

    h *= growy
    w *= growx
    if getattr(ax, "ax2", None) is None:
        ax.ax2 = plt.axes([x1 + offsetx*w, y1 + h - hist_h + gap/hf, w - hist_w, hist_h], sharex=ax, label=ax.get_label()+"_top")
        #ax.ax2.set_xticklabels([])
        ax.ax2.spines['right'].set_visible(False)
        ax.ax2.spines['top'].set_visible(False)
        ax.ax2.tick_params(axis='y', colors='none', which="both", labelcolor="none")
        ax.ax2.tick_params(axis='x', colors='none', which="both", labelcolor="none")
        ax.ax2.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.sca(ax.ax2)
    plot_density_hist(x, color=color, only_kde=only_kde)
    if getattr(ax, "ax3", None) is None:
        ax.ax3 = plt.axes([x1 + offsetx*w + w - hist_w + gap/wf, y1, hist_w, h - hist_h], sharey=ax, label=ax.get_label()+"_right")
        #ax.ax3.set_yticklabels([])
        ax.set_position([x1 + offsetx*w, y1, w - hist_w, h - hist_h])
        ax.ax3.spines['right'].set_visible(False)
        ax.ax3.spines['top'].set_visible(False)
        ax.ax3.tick_params(axis='x', colors='none', which="both", labelcolor="none")
        ax.ax3.tick_params(axis='y', colors='none', which="both", labelcolor="none")
        #ax.ax3.spines['left'].set_visible(False)
        ax.ax3.spines['bottom'].set_visible(False)

        ax.spines['right'].set_visible(False)
    plt.sca(ax.ax3)
    l = plot_density_hist(y, color=color, orientation=u'horizontal', only_kde=only_kde)
    plt.sca(ax)
    plotDensityLevels(x, y, levels=1, colors=[l.get_color()], cmap=None)
    plt.plot([], [], color=l.get_color(), label=label)


def get_cell_properties(data):
    import scipy.special
    from deformationcytometer.includes.RoscoeCoreInclude import getAlpha1, getAlpha2, getMu1, getEta1, eq41, getRoscoeStrain

    alpha1 = getAlpha1(data.long_axis / data.short_axis)
    alpha2 = getAlpha2(data.long_axis / data.short_axis)

    epsilon = getRoscoeStrain(alpha1, alpha2)

    mu1 = getMu1(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), data.stress)
    eta1 = getEta1(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), data.eta)

    if "omega" not in data:
        data["omega"] = data.stress*np.nan

    omega = np.abs(data.omega)

    ttfreq = - eq41(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), np.abs(data.vel_grad))
    #omega = ttfreq

    # omega = data.freq * 2 * np.pi

    def curve(x, x0, a):
        return 1 / 2 * 1 / (1 + (x / x0) ** a)

    omega_weissenberg = curve(np.abs(data.vel_grad), (1 / data.tau) * 3, data.delta) * np.abs(
        data.vel_grad)  # * np.pi*2
    #omega = omega_weissenberg

    Gp1 = mu1
    Gp2 = eta1 * np.abs(omega)
    alpha_cell = np.arctan(Gp2 / Gp1) * 2 / np.pi
    k_cell = Gp1 / (omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell))

    mu1_ = k_cell * omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell)
    eta1_ = k_cell * omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.sin(np.pi / 2 * alpha_cell) / omega

    data["omega"] = omega
    data["mu1"] = mu1
    data["eta1"] = eta1
    data["Gp1"] = Gp1
    data["Gp2"] = Gp2
    data["k_cell"] = k_cell
    data["alpha_cell"] = alpha_cell
    data["epsilon"] = epsilon


    w_Gp1 = mu1
    w_Gp2 = eta1 * np.abs(omega_weissenberg)
    w_alpha_cell = np.arctan(Gp2 / Gp1) * 2 / np.pi
    w_k_cell = Gp1 / (omega_weissenberg ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell))

    mu1_ = k_cell * omega_weissenberg ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell)
    eta1_ = k_cell * omega_weissenberg ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.sin(np.pi / 2 * alpha_cell) / omega_weissenberg

    data["w_Gp1"] = w_Gp1
    data["w_Gp2"] = w_Gp2
    data["w_k_cell"] = w_k_cell
    data["w_alpha_cell"] = w_alpha_cell

    return omega, mu1, eta1, k_cell, alpha_cell, epsilon


def match_cells_from_all_data(data, config, image_width=720):
    timestamps = {i: d.timestamp for i, d in data.groupby("frames").mean().iterrows()}
    for i, d in data.iterrows():
        x = d.x
        v = d.vel * 1e3 / config["pixel_size"]
        t = d.timestamp
        for dir in [1]:
            for j in range(1, 10):
                frame2 = d.frames + j * dir
                try:
                    dt = timestamps[frame2] - t
                except KeyError:
                    continue
                x2 = x + v * dt
                # if we ran out of the image, stop
                if not (0 <= x2 <= image_width):
                    break
                d2 = data[data.frames == frame2]
                # if the cell is already present in the frame, to not do any matching
                if len(d2[d2.cell_id == d.cell_id]):
                    continue
                d2 = d2[np.abs(d2.rp - d.rp) < 10]
                d2 = d2[np.abs(d2.x - x2) < 15]
                # if it survived the filters, merge
                if len(d2):
                    data.loc[data['cell_id'] == d2.iloc[0].cell_id, 'cell_id'] = d.cell_id


def split_axes(ax=None, join_x_axes=False, join_title=True):
    if ax is None:
        ax = plt.gca()
    x1, y1, w, h = ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height
    gap = w * 0.02
    if getattr(ax, "ax2", None) is None:
        ax.ax2 = plt.axes([x1 + w * 0.5 + gap, y1, w * 0.5 - gap, h], label=ax.get_label() + "_twin")
        ax.set_position([x1, y1, w * 0.5 - gap, h])
        # ax.ax2.set_xticklabels([])
        ax.ax2.spines['right'].set_visible(False)
        ax.ax2.spines['top'].set_visible(False)
        #ax.ax2.spines['left'].set_visible(False)
        ax.ax2.tick_params(axis='y', colors='gray', which="both", labelcolor="none")
        ax.ax2.spines['left'].set_color('gray')
        if join_title is True:
            ax.set_title(ax.get_title()).set_position([1.0 + gap, 1.0])
        if join_x_axes is True:
            t = ax.set_xlabel(ax.get_xlabel())
            t.set_position([1 + gap, t.get_position()[1]])
    plt.sca(ax.ax2)


def get_mode(x):
    """ get the mode of a distribution by fitting with a KDE """
    from scipy import stats
    x = np.array(x)
    x = x[~np.isnan(x)]

    kde = stats.gaussian_kde(x)
    return x[np.argmax(kde(x))]


def get_mode_stats(x, do_plot=False):
    from deformationcytometer.evaluation.helper_functions import bootstrap_error
    from scipy import stats

    x = np.array(x)
    x = x[~np.isnan(x)]

    def get_mode(x):
        kde = stats.gaussian_kde(x)
        return x[np.argmax(kde(x))]

    mode = get_mode(x)
    err = bootstrap_error(x, get_mode, repetitions=10)
    if do_plot is True:
        def string(x):
            if x > 1:
                return str(round(x))
            else:
                return str(round(x, 2))
        plt.text(0.5, 1, string(mode)+"$\pm$"+string(err), transform=plt.gca().transAxes, ha="center", va="top")
    return mode, err, len(x)



def bootstrap_match_hist(data_list, bin_width=25, max_bin=300, property="stress"):
    import pandas as pd
    # create empty lists
    data_list2 = [[] for _ in data_list]
    # iterate over the bins
    for i in range(0, max_bin, bin_width):
        # find the maximum
        counts = [len(data[(i < data[property]) & (data[property] < (i + bin_width))]) for data in data_list]
        max_count = np.max(counts)
        min_count = np.min(counts)
        # we cannot upsample from 0
        if min_count == 0:
            continue
        # iterate over datasets
        for index, data in enumerate(data_list):
            # get the subset of the data that is in this bin
            data_subset = data[(i < data[property]) & (data[property] < (i + bin_width))]
            # sample from this subset
            data_list2[index].append(data_subset.sample(max_count, replace=True))

    # concatenate the datasets
    for index, data in enumerate(data_list):
        data_list2[index] = pd.concat(data_list2[index])

    return data_list2

