import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import skimage.draw
from pathlib import Path
import pylustrator
import scipy.optimize
#pylustrator.start()

import sys
from scripts.helper_functions import getStressStrain, getConfig, getInputFile

def line(x, p):
    return x*p

def fitLine(x, y):
    def cost(p):
        #return np.mean((line(x, p)-y)**2)
        return np.mean(np.abs(line(x, p)-y))
    res = scipy.optimize.minimize(cost, [0.25])
    print(res, res["x"])
    return res["x"]

video = r"//131.188.117.96/biophysDS/emirzahossein/microfluidic cell rhemeter data/microscope_1/september_2020/2020_09_16_alginate2%_NIH_tanktreading/2/2020_09_16_14_34_28.tif"
#video = getInputFile()
print(video)

all_x = []
all_y = []

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

for video in list(Path(video).parent.glob("*.tif"))[::-1]:
    video = str(video)
    config = getConfig(video)

    target_folder = Path(video[:-4])

    data = pd.read_csv(target_folder / "output.csv")
    speeds = np.loadtxt(target_folder / "speeds.txt")

    getStressStrain(data, config)
    data["strain"] = (data.long_axis - data.short_axis)/np.sqrt(data.long_axis * data.short_axis)

    if speeds.shape == (0,):
        continue

    strains = [data[data.id == id].strain.mean() for id in speeds[:, 0]]
    stress = [data[data.id == id].stress.mean() for id in speeds[:, 0]]

    print(speeds)
    x, y = speeds[:, 1], -speeds[:, 2]#/(2*np.pi)


    #plt.subplot(121)
    for ax in [ax1, ax2]:
        plt.sca(ax)
        plt.plot(x, y, "o", ms=1, label=f'{config["pressure_pa"]*1e-5:3.1f}')


x2 = np.array([-70, 70])
#plt.plot(x2, 0.5*x2, "k--")
for ax in [ax1, ax2]:
    plt.sca(ax)
    plt.plot(x2, 0.25*x2, "k:")

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

mark_inset(ax1, ax2, loc1=2, loc2=3, fc="none", ec="0.5")

plt.legend()
#plt.loglog([])

plt.xlabel("shear rate (1/s)")
plt.ylabel("rotation frequency (1/s)")

plt.show()
