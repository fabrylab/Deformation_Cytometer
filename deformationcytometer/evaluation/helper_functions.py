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
        p0 = p0 * 1e3
        R = config["channel_width_m"] / 2 * 1e6
        return - (p0 * p1 * (np.abs(r) / R) ** p1) / r

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


def plotDensityScatter(x, y, cmap='viridis', alpha=1, skip=1, y_factor=1, levels=None):
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
    plt.scatter(x, y, c=z, s=5, alpha=alpha, cmap=cmap)  # plot in kernel density colors e.g. viridis

    if levels != None:
        X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
        print(xy.shape)
        print(np.dstack([X, Y*y_factor]).shape)
        XY = np.dstack([X, Y*y_factor])
        Z = kde(XY.reshape(-1, 2).T).reshape(XY.shape[:2])
        plt.contour(X, Y, Z, levels=1)

    #

def plotDensityLevels(x, y, skip=1, y_factor=1, levels=None, cmap="viridis"):
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
    plt.contour(X, Y, Z, levels=levels, cmap=cmap)

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


def bootstrap_error(data, func=np.median):
    data = np.asarray(data)
    if len(data) <= 1:
        return 0
    medians = []
    for i in range(1000):
        medians.append(func(data[np.random.random_integers(len(data) - 1, size=len(data))]))
    return np.nanstd(medians)


def plotBinnedData(x, y, bins, bin_func=np.median, error_func=None, color="black", mew=1, mfc='white'):
    x = np.asarray(x)
    y = np.asarray(y)
    strain_av = []
    stress_av = []
    strain_err = []
    for i in range(len(bins) - 1):
        index = (bins[i] < x) & (x < bins[i + 1])
        yy = y[index]
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
    plt.errorbar(stress_av, strain_av, yerr=np.array(strain_err).T, marker='s', mfc=mfc, \
                 mec=color, ms=7, mew=mew, lw=0, ecolor='black', elinewidth=1, capsize=3)
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

    evaluation_version = 6

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

        version = 0
        if output_config_file.exists():
            with output_config_file.open("r") as fp:
                config = json.load(fp)
                config["channel_width_m"] = 0.00019001261833616293
            if "evaluation_version" in config:
                version = config["evaluation_version"]
        if "filter" in config.keys():
            filters.append(config["filter"])

        """ evaluating data"""
        if not output_file.exists() or version < evaluation_version:
            #refetchTimestamps(data, config)

            getVelocity(data, config)
            # take the mean of all values of each cell
            data = data.groupby(['cell_id']).mean()

            correctCenter(data, config)

            data = filterCells(data, config)
            # reset the indices
            data.reset_index(drop=True, inplace=True)

            getStressStrain(data, config)

            #data = data[(data.stress < 50)]
            data.reset_index(drop=True, inplace=True)

            data["area"] = data.long_axis * data.short_axis * np.pi
            data["pressure"] = config["pressure_pa"]*1e-5

            data, p = apply_velocity_fit(data)

            try:
                config["evaluation_version"] = evaluation_version
                data.to_csv(output_file, index=False)
                #print("config", config, type(config))
                with output_config_file.open("w") as fp:
                    json.dump(config, fp)

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
    data = pd.concat(data_list)
    data.reset_index(drop=True, inplace=True)

    #fitStiffness(data, config)
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

    for pressure in sorted(data.pressure.unique(), reverse=True):
        d = data[data.pressure == pressure]
        d = d.set_index(["eta0", "delta", "tau"])
        for p in d.index.unique():
            dd = d.loc[p]
            x, y = getFitLine(pressure, p)
            line, = plt.plot(np.abs(dd.rp), dd.velocity * 1e-3 * 1e2, "o", alpha=0.3, ms=2, color=color)
            plt.plot([], [], "o", ms=2, color=line.get_color(), label=f"{pressure:.1f}")
            l, = plt.plot(x[x>=0]* 1e+6, y[x>=0] * 1e2, color="k")
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

def plot_density_hist(x, **kwargs):
    x = np.array(x)
    from scipy import stats
    kde = stats.gaussian_kde(x[~np.isnan(x)])
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 1000)
    l, = plt.plot(xx, kde(xx), **kwargs)
    plt.hist(x, bins=50, density=True, color=l.get_color(), alpha=0.5)

def get_cell_properties(data):
    import scipy.special
    from deformationcytometer.includes.RoscoeCoreInclude import getAlpha1, getAlpha2, getMu1, getEta1, eq41, getRoscoeStrain

    alpha1 = getAlpha1(data.long_axis / data.short_axis)
    alpha2 = getAlpha2(data.long_axis / data.short_axis)

    epsilon = getRoscoeStrain(alpha1, alpha2)

    mu1 = getMu1(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), data.stress)
    eta1 = getEta1(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), data.eta)

    ttfreq = - eq41(alpha1, alpha2, np.abs(np.deg2rad(data.angle)), np.abs(data.vel_grad))
    omega = ttfreq
    #omega = data.freq * 2 * np.pi

    def curve(x, x0, a):
        return 1 / 2 * 1 / (1 + (x / x0) ** a)

    omega_weissenberg = curve(np.abs(data.vel_grad), (1 / data.tau) * 3, data.delta) * np.abs(data.vel_grad)  # * np.pi*2
    omega = omega_weissenberg

    Gp1 = mu1
    Gp2 = eta1 * np.abs(omega)
    alpha_cell = np.arctan(Gp2 / Gp1) * 2 / np.pi
    k_cell = Gp1 / (omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell))

    mu1_ = k_cell * omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.cos(np.pi / 2 * alpha_cell)
    eta1_ = k_cell * omega ** alpha_cell * scipy.special.gamma(1 - alpha_cell) * np.sin(np.pi / 2 * alpha_cell) / omega

    return omega, mu1, eta1, k_cell, alpha_cell, epsilon