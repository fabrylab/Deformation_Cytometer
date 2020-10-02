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

#import pylustrator
#pylustrator.start()

""" loading data """
# get the results file (by config parameter or user input dialog)
datafiles = [
    #r"\\131.188.117.96\biophysDS\emirzahossein\data\microscope4_baslercamera\2020_may\2020_05_22_alginateDMEM2%\3\2020_05_22_10_09_10_result.txt",
    #r"\\131.188.117.96\biophysDS\emirzahossein\data\microscope4_baslercamera\2020_july\2020_07_23_alginate2%_3t3_margination\1bar\2020_07_23_10_07_42_result.txt",
    r"\\131.188.117.96\biophysDS\emirzahossein\data\microscope4_baslercamera\2020_july\2020_07_23_alginate2%_3t3_margination\2bar\2020_07_23_09_59_11_result.txt"
#    r"\\131.188.117.96\biophysDS\emirzahossein\data\microscope4_baslercamera\2020_july\2020_07_23_alginate2%_3t3_margination\3bar\2020_07_23_10_05_02_result.txt"
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

    getStressStrain(data, config)

    fitStiffness(data, config)

    error = np.abs(config["fit"]["fitfunc"](data.stress, *config["fit"]["p"]) - data.strain)
    data = data[error < 0.15]
    data.reset_index(drop=True, inplace=True)

    data["area"] = data.long_axis * data.short_axis * np.pi
    r = np.sqrt(data.area/np.pi)

    data = data[np.abs(r-np.mean(r)) < 0.5]
    data.reset_index(drop=True, inplace=True)

    # reset the indices
    data.reset_index(drop=True, inplace=True)
    datas.append(data)

width = 100
height = 70
scale = 0.25

xpos = 0
for im, data in zip(image_stacks, datas):
    for i in range(100):
        xpos += width*scale + 5

        ypos = 75
        for ypos in [-75, -50, -25, 0, 25, 50, 75]:
            #plt.axhline(ypos, linestyle="--", color="k", lw=0.8)
            nearest_id = np.argsort(np.abs(data.rp - ypos))[i]
            print(nearest_id)

            d = data.iloc[nearest_id]

            image = im.get_data(d.frames)
            image = image[int(d.y-height//2):int(d.y+height//2), int(d.x-width/2):int(d.x+width/2)]
            plt.text(xpos, ypos, f"{nearest_id}\n{d.rp:.2f}")

            plt.imshow(image, cmap="gray", extent=[xpos, xpos+image.shape[1]*scale, ypos+image.shape[0]*scale/2, ypos-image.shape[1]*scale/2])

plt.axis("equal")
#plt.savefig(__file__[:-3]+".png", dpi=300)
#plt.savefig(__file__[:-3]+".pdf")
plt.show()