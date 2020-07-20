import sys
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_functions import getInputFile, getConfig, getData
from helper_functions import refetchTimestamps, getVelocity, filterCells, correctCenter, getStressStrain, fitStiffness
from helper_functions import initPlotSettings, plotVelocityProfile, plotStressStrain, plotMessurementStatus
from helper_functions import storeEvaluationResults

import pylustrator
pylustrator.start()

""" loading data """
# get the results file (by config parameter or user input dialog)
datafiles = [
    r"\\131.188.117.96\biophysDS\emirzahossein\data\microscope4_baslercamera\2020_may\2020_05_22_alginateDMEM2%\3\2020_05_22_10_09_10_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\data\microscope4_baslercamera\2020_may\2020_05_22_alginateDMEM2%\3\2020_05_22_10_10_43_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\data\microscope4_baslercamera\2020_may\2020_05_22_alginateDMEM2%\3\2020_05_22_10_11_38_result.txt",
]

datas = []
image_stacks = []

for datafile in datafiles:
    #datafile = getInputFile(filetype=[("txt file",'*_result.txt')])

    # load the data and the config
    data = getData(datafile)
    config = getConfig(datafile)
    print(config)

    im = imageio.get_reader(datafile.replace("_result.txt", ".tif"))
    image_stacks.append(im)

    getVelocity(data, config)

    # take the mean of all values of each cell
    #data = data.groupby(['cell_id']).mean()

    correctCenter(data, config)

    data = filterCells(data, config)

    # reset the indices
    data.reset_index(drop=True, inplace=True)
    datas.append(data)

width = 100
height = 70
scale = 0.25

xpos = 0
for im, data in zip(image_stacks, datas):
    xpos += width*scale + 5

    ypos = 75
    for ypos in [-75, -50, -25, 0, 25, 50, 75]:
        plt.axhline(ypos, linestyle="--", color="k", lw=0.8)
        nearest_id = np.argsort(np.abs(data.rp - ypos))[0]
        print(nearest_id)

        d = data.iloc[nearest_id]

        image = im.get_data(d.frames)
        image = image[int(d.y-height//2):int(d.y+height//2), int(d.x-width/2):int(d.x+width/2)]

        plt.imshow(image, cmap="gray", extent=[xpos, xpos+image.shape[1]*scale, ypos+image.shape[0]*scale/2, ypos-image.shape[1]*scale/2])

plt.axis("equal")
plt.xticks([])
plt.yticks([-100, -75, -50, -25, 0, 25, 50, 75, 100],
           ["+100", "+75", "+50", "+25", "0", "-25", "-50", "-75", "-100"])
plt.figure(1).axes[0].spines['left'].set_visible(False)
plt.figure(1).axes[0].spines['right'].set_visible(False)
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(7.980000/2.54, 9.670000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.217521, 0.026664, 0.667013, 0.943955])
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[0].new
plt.figure(1).axes[0].texts[0].set_ha("center")
plt.figure(1).axes[0].texts[0].set_position([0.240994, 0.960917])
plt.figure(1).axes[0].texts[0].set_text("1 bar")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[1].new
plt.figure(1).axes[0].texts[1].set_ha("center")
plt.figure(1).axes[0].texts[1].set_position([0.500000, 0.960917])
plt.figure(1).axes[0].texts[1].set_text("2 bar")
plt.figure(1).axes[0].text(0.5, 0.5, 'New Text', transform=plt.figure(1).axes[0].transAxes)  # id=plt.figure(1).axes[0].texts[2].new
plt.figure(1).axes[0].texts[2].set_ha("center")
plt.figure(1).axes[0].texts[2].set_position([0.754297, 0.960917])
plt.figure(1).axes[0].texts[2].set_text("3 bar")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("radial position (µm)")
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()