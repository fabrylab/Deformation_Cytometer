import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import skimage.draw
from pathlib import Path
import pylustrator
import scipy.optimize
pylustrator.start()

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

video = getInputFile()

config = getConfig(video)

target_folder = Path(video[:-4])

data = pd.read_csv(target_folder / "output.csv")
speeds = np.loadtxt(target_folder / "speeds.txt")

getStressStrain(data, config)
data["strain"] = (data.long_axis - data.short_axis)/np.sqrt(data.long_axis * data.short_axis)

strains = [data[data.id == id].strain.mean() for id in speeds[:, 0]]
stress = [data[data.id == id].stress.mean() for id in speeds[:, 0]]
if 0:
    #data = data.groupby(['id']).mean()
    plt.plot(stress, strains, "o")

    sigma = stress#np.arange(0, 50)
    for alpha in [0.2, 0.3, 0.4, 0.5]:
        k = 3
        #alpha = 0.5
        gamma = np.arange(0, 250)
        w = np.abs(-speeds[:, 1])
        w = gamma/4
        epsilon = gamma / (k * 2*np.pi*w**alpha) #* np.sqrt(1- (np.tan(np.pi/2 * alpha))/(k*w**(alpha+1)))
        print(epsilon)
        p, = plt.plot(gamma, epsilon, "o")

        epsilon = gamma / (k * 2 * np.pi * w ** alpha) * np.sqrt(1- ((np.tan(np.pi/2 * alpha))/(k*w**(alpha+1)))**2)
        print(epsilon)
        plt.plot(gamma, epsilon, "d", color=p.get_color())
    plt.show()
    sys.exit()

print(speeds)
x, y = speeds[:, 1], -speeds[:, 2]#/(2*np.pi)

indices = ~(np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y))
y = y[indices]
x = x[indices]

fitLine(x, y)

t = 0
m = 0.5
plt.title(config["pressure_pa"]*1e-5)
#plt.subplot(121)
plt.plot(x, y, "o", ms=1)
x2 = np.array([np.min(x), np.max(x)])
plt.plot(x2, m*x2+t, "k--")
plt.plot(x2, 0.25*x2+t, "k:")

m = scipy.optimize.curve_fit(line, x, y)[0]
print(m)
#plt.plot(x2, m*x2+t, "r--")

m = np.mean(y/x)
print(m)
#plt.plot(x2, m*x2+t, "m--")

x2 = np.abs(x)
y2 = np.abs(y)
bins = np.arange(0, 50, 2.5)
binned = [np.median(y2[(bins[i] < x2) & (x2 < bins[i+1])]) for i in range(len(bins)-1)]
binned_std = [np.std(y2[(bins[i] < x2) & (x2 < bins[i+1])]) for i in range(len(bins)-1)]
binned_x = bins[:-1]+np.diff(bins)[0]/2
#plt.errorbar(binned_x, binned, yerr=binned_std, zorder=10)
#plt.plot(x2, y2, "+")
#plt.axis("equal")


plt.xlabel("shear rate (1/s)")
plt.ylabel("rotation frequency (1/s)")
if 0:
    plt.subplot(122)
    #plt.plot(np.abs(speeds[:, 2]*2), np.abs(-speeds[:, 1]), "o")
    plt.plot(strains, np.abs(-speeds[:, 1]), "o")
    x2 = np.arange(np.min(x), np.max(x))
    plt.plot(x2, m*x2+t, "k--")
    plt.xlabel("shear rate (1/s)")
    plt.ylabel("rotation frequency (1/s)")

plt.savefig(video[:-4]+"_rotation.png")
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(8.000000/2.54, 8.000000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.201433, 0.183248, 0.698567, 0.696752])
plt.figure(1).axes[0].set_xlim(-50.0, 50.0)
plt.figure(1).axes[0].set_ylim(-20.0, 20.0)
plt.figure(1).axes[0].title.set_text("")
#% end: automatic generated code from pylustrator
plt.show()
