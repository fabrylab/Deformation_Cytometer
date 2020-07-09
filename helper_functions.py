import numpy as np
import copy
from tkinter import Tk
from tkinter import filedialog
import sys
import os
from pathlib import Path
import configparser
import imageio
import cv2
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde


def getInputFile(filetype=[("video file",'*.tif *.avi')]):
    # if there is a command line parameter...
    if len(sys.argv) >= 2:
        # ... we just use this file
        video = sys.argv[1]
    # if not, we ask the user to provide us a filename
    else:
        # select video file
        root = Tk()
        root.withdraw() # we don't want a full GUI, so keep the root window from appearing
        video = []
        video = filedialog.askopenfilename(title="select the data file",filetypes=filetype) # show an "Open" dialog box and return the path to the selected file
        if video == '':
            print('empty')
            sys.exit()
    return video


def getInputFolder():
    # if there are command line parameters, we use the provided folder
    if len(sys.argv) >= 2:
        parent_folder = sys.argv[1]
    # if not we ask for a folder
    else:
        #%% select video file
        root = Tk()
        root.withdraw() # we don't want a full GUI, so keep the root window from appearing
        parent_folder = []
        parent_folder = filedialog.askdirectory(title="select the parent folder") # show an "Open" dialog box and return the path to the selected file
        if parent_folder == '':
            print('empty')
            sys.exit()
    return parent_folder


#%% open and read the config file
def getConfig(configfile):
    if configfile.endswith("_result.txt"):
        configfile = configfile.replace("_result.txt", "_config.txt")
    if configfile.endswith(".tif"):
        configfile = configfile.replace(".tif", "_config.txt")


    config = configparser.ConfigParser()
    config.read(configfile)

    config_data = {}
    print("config", config, configfile)

    config_data["file_data"] = configfile.replace("_config.txt", "_result.txt")
    config_data["file_tif"] = configfile.replace("_config.txt", ".tif")
    config_data["file_config"] = configfile

    config_data["magnification"] = float(config['MICROSCOPE']['objective'].split()[0])
    config_data["coupler"] = float(config['MICROSCOPE']['coupler'] .split()[0])
    config_data["camera_pixel_size"] = float(config['CAMERA']['camera pixel size'] .split()[0])
    config_data["pixel_size"] = config_data["camera_pixel_size"]/(config_data["magnification"]*config_data["coupler"]) # in meter
    config_data["px_to_um"] = config_data["pixel_size"]
    config_data["pixel_size_m"] = config_data["pixel_size"] * 1e-6 # in um
    config_data["channel_width_px"] = float(config['SETUP']['channel width'].split()[0])*1e-6/config_data["pixel_size"] #in pixels

    config_data["pressure_pa"] = float(config['SETUP']['pressure'].split()[0]) * 1000  # applied pressure (in Pa)

    config_data["channel_width_m"] = float(config['SETUP']['channel width'].split()[0])*1e-6
    config_data["channel_length_m"] = float(config['SETUP']['channel length'].split()[0])*1e-2

    return config_data

def getData(datafile):
    # %% import raw data
    data = np.genfromtxt(datafile, dtype=float, skip_header=2)
    data = pd.DataFrame({
        "frames": data[:, 0].astype(int),
        "x": data[:, 1],
        "y": data[:, 2],
        "rp": data[:, 3],
        "long_axis": data[:, 4],
        "short_axis": data[:, 5],
        "angle": data[:, 6],
        "irregularity": data[:, 7],
        "solidity": data[:, 8],
        "sharpness": data[:, 9],
        "timestamp": data[:, 10],
    })
    return data

#%%  compute average (flatfield) image
def getFlatfield(video, flatfield, force_recalculate=False):
    if os.path.exists(flatfield) and not force_recalculate:
        im_av = np.load(flatfield)
    else:
        vidcap = imageio.get_reader(video)
        print("compute average (flatfield) image")
        count = 0
        progressbar = tqdm.tqdm(vidcap)
        progressbar.set_description("computing flatfield")
        for image in progressbar:
            if len(image.shape) == 3:
                image = image[:,:,0]
            if count == 0:
                im_av = copy.deepcopy(image)
                im_av = np.asarray(im_av)
                im_av.astype(float)
            else:
                im_av = im_av + image.astype(float)
            count += 1
        im_av = im_av / count
        try:
            np.save(flatfield, im_av)
        except PermissionError as err:
            print(err)
    return im_av


def convertVideo(input_file, output_file=None, rotate=True):
    if output_file is None:
        basefile, extension = os.path.splitext(input_file)
        new_input_file = basefile + "_raw" + extension
        os.rename(input_file, new_input_file)
        output_file = input_file
        input_file = new_input_file

    if input_file.endswith(".tif"):
        vidcap = imageio.get_reader(input_file)
        video = imageio.get_writer(output_file)
        count = 0
        for im in vidcap:
            print(count)
            if len(im.shape) == 3:
                im = im[:,:,0]
            if rotate:
                im = im.T
                im = im[::-1,::]

            video.append_data(im)
            count += 1
        return

    vidcap = cv2.VideoCapture(input_file)
    video = imageio.get_writer(output_file, quality=7)
    count = 0
    success = True
    while success:
        success,im = vidcap.read()
        print(count)
        if success:
            if len(im.shape) == 3:
                im = im[:,:,0]
            if rotate:
                im = im.T
                im = im[::-1,::]

            video.append_data(im)
            count += 1


def stressfunc(R, P, L, H): # imputs (radial position and pressure)
    print("stressfunc", np.max(R), P, L, H)
    G = P/L  # pressure gradient
    pre_factor = (4*(H**2)*G)/np.pi**3
    # sum only over odd numbers
    n = np.arange(1, 100, 2)[None, :]
    u_primy = np.sum(pre_factor * ((-1) ** ((n - 1) / 2)) * (np.pi / ((n ** 2) * H)) \
                        * (np.sinh((n * np.pi * R[:, None]) / H) / np.cosh(n * np.pi / 2)), axis=1)

    stress = np.abs(u_primy)
    return stress


def refetchTimestamps(data, config):
    import json
    import imageio
    def getTimestamp(vidcap, image_index):
        if vidcap.get_meta_data(image_index)['description']:
            return json.loads(vidcap.get_meta_data(image_index)['description'])['timestamp']
        return "0"

    vidcap = imageio.get_reader(config["file_tif"])

    timestamp = data.timestamp.to_numpy()
    for index in data.index:
        timestamp[index] = float(getTimestamp(vidcap, int(data.frames[index] + 1)))
    data.timestamp = timestamp


def getVelocity(data, config):
    # %% compute velocity profile
    y_pos = []
    vel = []
    velocities = np.zeros(data.shape[0])*np.nan
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
                    velocity_partner[i] = f"{i}, {i+j}, {dt}, {data.x[i + j] - data.x[i]}, {data.frames[i]}, {data.long_axis[i]}, {data.short_axis[i]} -> {data.frames[i+j]}, {data.long_axis[i+j]}, {data.short_axis[i+j]}"
                    cell_id[i+j] = cell_id[i]
    data["velocity"] = velocities
    data["velocity_partner"] = velocity_partner
    data["cell_id"] = cell_id


def getStressStrain(data, config):
    data["stress"] = stressfunc(data.rp*1e-6, -config["pressure_pa"], config["channel_length_m"], config["channel_width_m"])
    data["strain"] = (data.long_axis - data.short_axis)/np.sqrt(data.long_axis * data.short_axis)

def filterCells(data, config):
    l_before = data.shape[0]
    data = data[(data.solidity > 0.96) & (data.irregularity < 1.05) & (data.rp.abs() < 65)]

    l_after = data.shape[0]
    print('# frames =', data.frames.iloc[-1], '   # cells total =', l_before, '   #cells sorted = ', l_after)
    print('ratio #cells/#frames before sorting out = %.2f \n' % float(l_before / data.frames.iloc[-1]))

    config["filter"] = dict(l_before=l_before, l_after=l_after)

    data.reset_index(drop=True, inplace=True)

    return data

def correctCenter(data, config):
    if not "velocity" in data:
        getVelocity(data, config)
    d = data[~np.isnan(data.velocity)]
    y_pos = d.rp
    vel = d.velocity

    def velfit(r, p0, p1, p2):  # for stress versus strain
        R = config["channel_width_m"] / 2 * 1e6
        return p0 * (1 - np.abs((r + p2) / R) ** p1)

    vel_fit, pcov = curve_fit(velfit, y_pos, vel, [np.max(vel), 0.9, 0])  # fit a parabolic velocity profile
    y_pos += vel_fit[2]
    #data.y += vel_fit[2]
    data.rp += vel_fit[2]

    config["vel_fit"] = vel_fit
    config["center"] = vel_fit[2]

    """  
    #%% find center of the channel 
    no_right_cells = 0
    center = 0
    for i in np.arange(-50,50,0.1):
        n = np.sum(np.sign(-(data.rp+i)*data.angle))
        if n>no_right_cells:
            center = i
            no_right_cells = n
    print('center channel position at y = %.1f  \u03BCm' % -center)
    data.rp = data.rp + center
    if np.max(data.rp)> 1e6*config["channel_width_m"]/2:
        data.rp = data.rp - (np.max(data.rp)-1e6*config["channel_width_m"]/2)  #this is to ensure that the maximum or minimum radial position
    if np.min(data.rp) < -1e6*config["channel_width_m"]/2:           #of a cell is not outsied the channel
        data.rp = data.rp - (np.min(data.rp)+1e6*config["channel_width_m"]/2)
    #y_pos = y_pos+center
    """

def fitStiffness(data, config):

    def fitfunc(x, p0, p1, p2):  # for stress versus strain
        return (1 / p0) * np.log((x / p1) + 1) + p2

    pstart = (0.1, 1, 0)  # initial guess

    xy = np.vstack([data.stress, data.strain])
    kd = gaussian_kde(xy)(xy)

    # fit weighted by the density of points
    p, pcov = curve_fit(fitfunc, data.stress, data.strain, pstart, sigma=1 / kd, maxfev=10000)  # do the curve fitting
    # p, pcov = curve_fit(fitfunc, stress[RP<0], strain[RP<0], pstart) #do the curve fitting for one side only
    err = (np.diag(pcov)) ** 0.5  # estimate 1 standard error of the fit parameters
    cov_ap = pcov[0, 1]  # cov between alpha and prestress
    cov_ao = pcov[0, 2]  # cov between offset and alpha
    cov_po = pcov[1, 2]  # cov between prestress and offset
    se01 = np.sqrt((p[1] * err[0]) ** 2 + (p[0] * err[1]) ** 2 + 2 * p[0] * p[1] * cov_ap)
    print('pressure = %5.1f kPa' % float(config["pressure_pa"] / 1000))
    print("p0 =%5.2f   p1 =%5.1f Pa   p0*p1=%5.1f Pa   p2 =%4.3f" % (p[0], p[1], p[0] * p[1], p[2]))

    print("se0=%5.2f   se1=%5.1f Pa   se0*1=%5.1f Pa   se2=%4.3f" % (err[0], err[1], err[0] * err[1], err[2]))

    config["fit"] = dict(fitfunc=fitfunc, p=p, err=err, cov_ap=cov_ap, cov_ao=cov_ao, cov_po=cov_po)

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
    ax1.set_ylim(0, 1.1*vel_fit[0])

    r = np.arange(-config["channel_width_m"] / 2 * 1e6, config["channel_width_m"] / 2 * 1e6,
                  0.1)  # generates an extended array

    ax1.plot(-y_pos, vel, '.', mfc="none", mec="C1", alpha=0.5)
    ax1.plot(y_pos, vel, '.', color="C0")
    ax1.plot(r, velfit(r, vel_fit[0], vel_fit[1]), '--', color='gray', linewidth=2, zorder=3)
    ax1.axvline(0, ls="--", color="k", lw=0.8)
    ax1.axhline(0, ls="--", color="k", lw=0.8)
    print('v_max = %5.2f mm/s   profile stretch exponent = %5.2f\n' % (vel_fit[0], vel_fit[1]))

def plotDensityScatter(x, y):
    xy = np.vstack([x, y])
    kd = gaussian_kde(xy)(xy)
    idx = kd.argsort()
    x, y, z = x[idx], y[idx], kd[idx]
    plt.scatter(x, y, c=z, s=50, edgecolor='', alpha=1, cmap='viridis')  # plot in kernel density colors e.g. viridis


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
    plt.plot(xx, (fitfunc(xx, p[0], p[1], p[2])), '-', color='black', linewidth=2, zorder=3)

    # ----------plot standard error of the fit function----------
    dyda = -1 / (p[0] ** 2) * np.log(xx / p[1] + 1)  # strain derivative with respect to alpha
    dydp = -1 / p[0] * xx / (xx * p[1] + p[1] ** 2)  # strain derivative with respect to prestress
    dydo = 1  # strain derivative with respect to offset
    vary = (dyda * err[0]) ** 2 + (dydp * err[1]) ** 2 + (
                dydo * err[2]) ** 2 + 2 * dyda * dydp * cov_ap + 2 * dyda * dydo * cov_ao + 2 * dydp * dydo * cov_po
    y1 = fitfunc(xx, p[0], p[1], p[2]) - np.sqrt(vary)
    y2 = fitfunc(xx, p[0], p[1], p[2]) + np.sqrt(vary)
    plt.fill_between(xx, y1, y2, facecolor='gray', edgecolor="none", linewidth=0, alpha=0.5)


def plotBinnedData(x, y, bins):
    strain_av = []
    stress_av = []
    strain_err = []
    for i in range(len(bins) - 1):
        index = (x > bins[i]) & (x < bins[i + 1])
        strain_av.append(np.mean(y[index]))
        strain_err.append(np.std(y[index]) / np.sqrt(np.sum(index)))
        stress_av.append(np.mean(x[index]))
    plt.errorbar(stress_av, strain_av, yerr=strain_err, marker='s', mfc='white', \
                 mec='black', ms=7, mew=1, lw=0, ecolor='black', elinewidth=1, capsize=3)


def plotStressStrain(data, config):
    # %% fitting deformation with stress stiffening equation
    fig2 = plt.figure(2, (6, 6))
    border_width = 0.1
    ax_size = [0 + 2 * border_width, 0 + 2 * border_width,
               1 - 3 * border_width, 1 - 3 * border_width]
    ax2 = fig2.add_axes(ax_size)
    ax2.set_xlabel('fluid shear stress $\u03C3$ (Pa)')
    ax2.set_ylabel('cell strain  $\u03B5$')

    pmax = 50 * np.ceil((np.max(data.stress) + 50) // 50)
    ax2.set_xticks(np.arange(0, pmax + 1, 50))
    ax2.set_xlim((-10, pmax))
    ax2.set_ylim((-0.2, 1.0))

    # ----------plot strain versus stress data points----------
    plotDensityScatter(data.stress, data.strain)

    # ----------plot the fit curve----------
    plotStressStrainFit(data, config)

    # ----------plot the binned (averaged) strain versus stress data points----------
    plotBinnedData(data.stress, data.strain, [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250])

def plotMessurementStatus(data, config):
    firstPage = plt.figure(figsize=(11.69, 8.27))
    firstPage.clf()
    txt = '# frames =', data.frames.iloc[-1], '   # cells total =', config["filter"]["l_before"], '   #cells sorted = ', config["filter"]["l_after"]
    txt1 = 'ratio #cells/#frames before sorting out = %.2f \n' % float(config["filter"]["l_before"] / data.frames.iloc[-1])
    txt2 = 'center channel position at y = %.1f  \u03BCm' % -config["center"]
    txt3 = 'v_max = %5.2f mm/s   profile stretch exponent = %5.2f\n' % (config["vel_fit"][0], config["vel_fit"][1])
    txt4 = 'pressure = %5.1f kPa' % float(config["pressure_pa"] / 1000)

    fit = config["fit"]
    p = fit["p"]
    err = fit["err"]
    txt5 = "p0 =%5.2f   p1 =%5.1f Pa   p0*p1=%5.1f Pa   p2 =%4.3f" % (p[0], p[1], p[0] * p[1], p[2])
    txt6 = "se0=%5.2f   se1=%5.1f Pa   se0*1=%5.1f Pa   se2=%4.3f" % (err[0], err[1], err[0] * err[1], err[2])
    txt_whole = str(txt) + "\n" + str(txt1) + "\n" + str(txt2) + "\n" + str(txt3) + "\n" + str(txt4) + "\n" + str(
        txt5) + "\n" + str(txt6)
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
    f.write('{:0.3f}'.format(config["fit"]["p"][0]) + '\t' + '{:0.2f}'.format(config["fit"]["p"][1]) + '\t' + '{:0.3f}'.format(
        config["fit"]["p"][2]) + '\t' + '{:0.3f}'.format(config["fit"]["p"][0] * config["fit"]["p"][1]) + '\n')
    # f.write(str(frame[i]) +'\t' +str(X[i]) +'\t' +str(Y[i]) +'\t' +str(R[i]) +'\t' +str(LongAxis[i]) +'\t'+str(ShortAxis[i]) +'\t' +str(Angle[i]) +'\t' +str(irregularity[i]) +'\t' +str(solidity[i]) +'\t' +str(sharpness[i]) +'\n')
    f.close()