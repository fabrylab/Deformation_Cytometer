import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data, load_all_data_new, bootstrap_match_hist
import numpy as np
import pylustrator
import glob

if 0:
    with open("figure_error.txt", "w") as fp:
        for file in glob.glob(r"\\131.188.117.96\biophysDS\emirzahossein\**\*.tif", recursive=True):
            print(file)
            fp.write(file+"\n")
        for file in glob.glob(r"\\131.188.117.96\biophysDS\meroles\**\*.tif", recursive=True):
            print(file)
            fp.write(file+"\n")
if 0:
    files = {}
    with open("figure_error.txt", "r") as fp:
        for line in fp:
            if "microscope4" in line or "microscope_1" in line or "meroles" in line:
                line = line.strip()
                print(Path(line).name)
                files[Path(line).name] = line
    #exit()
    import pickle
    with open("figure_error.pickle", "wb") as fp:
        pickle.dump(files, fp)
import pickle
with open("figure_error.pickle", "rb") as fp:
    files = pickle.load(fp)
print(files)
#pylustrator.start()
import clickpoints
data = []
db_files = ["gt_0_selina.cdb"]
for index in [0, 2, 3, 4, 5, 6, 7, 8]:
    data1, config = load_all_data_new(fr"\\131.188.117.96\biophysDS\emirzahossein\groundtruth_data\data\outline_GT\gt_{index}_evaluated_evaluated_new.csv", do_group=False)
    data2, config = load_all_data_new(fr"\\131.188.117.96\biophysDS\emirzahossein\groundtruth_data\data\outline_GT\gt_{index}_evaluated_copy_evaluated_new.csv", do_group=False)
    if 0:
        with clickpoints.DataFile(r"\\131.188.117.96\biophysDS\emirzahossein\groundtruth_data\data\outline_GT\\"+db_files[index]) as cdb:
            for image in cdb.getImageIterator():
                filename = cdb.getMarkers(image, type="image")[0].text
                if filename not in files:
                    continue
                file = files[filename]
                d = load_all_data_new(file.replace(".tif", "_evaluated_new.csv"))[0].iloc[0]
                data1.loc[data1.frames == image.sort_index, "pressure"] = d.pressure


    for i, d in data1.iterrows():
        d_ = data2[data2.frames == d.frames]
        d_ = d_[np.abs(d_.x - d.x) < 5]
        d_ = d_[np.abs(d_.y - d.y) < 5]
        if len(d_):
            d_ = d_.iloc[0]
            print(d)
            data.append([d.angle, d_.angle, d.x, d_.x, d.frames, d.rp])

data = np.array(data)
print(len(data))
print(data)
#plt.plot(data[:, 0], data[:, 0]-data[:, 1], "o", ms=1)
plt.plot(data[:, 0], data[:, 0]-data[:, 1], "o", ms=1)
#plt.plot(data1.frames, data1.x)
#plt.plot(data2.frames, data2.x)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(12.000000/2.54, 6.000000/2.54, forward=True)
#% end: automatic generated code from pylustrator
plt.savefig(__file__[:-3]+".png", dpi=300)
plt.savefig(__file__[:-3]+".pdf")
plt.show()
