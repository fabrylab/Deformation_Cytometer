import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data_new, get2Dhist_k_alpha

# load all the data in the given folder
# you can also add a list of folders instead
# filenames can also include * for wildcard matches

data, config = load_all_data_new([
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_1",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_18_alginate2%_overtime_2",
        rf"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\august_2020\2020_08_19_alginate2%_overtime_1",
    ])

# print all the columns, these are the ones from the results.csv and from the meta data files
print(data.columns)

data = data.query("1 < pressure < 2 and treatment == 'DMSO'")

data = data[data.treatment == "DMSO"]

def plot_over_time():
    groupped_data = data.groupby(["measurement_id", "time"]).apply(get2Dhist_k_alpha)
    groupped_data.reset_index("time", inplace=True)

    # the mean curve
    groupped_2 = groupped_data.groupby("time")["k"].agg(["mean", "sem", "count"])
    l = plt.errorbar(groupped_2.index, groupped_2["mean"].values, yerr=groupped_2["sem"].values, capsize=3, zorder=2)
    print("count", "time", "k", groupped_2["count"].values, groupped_2["mean"].values)

    # the individual curves
    for id, d2 in groupped_data.groupby("measurement_id"):
        plt.plot(d2["time"], d2["k"], "o", ms=3, color=l[0].get_color(), alpha=0.5)
        plt.plot(d2["time"], d2["k"], "-", ms=3, color="gray", alpha=0.5)
    plt.xlabel("time")
    plt.ylabel("k")

# you can now group by some of the columns
# for example just by the filename to get an evaluation per file
for filename, d in data.groupby("filename"):
    # calculate the k and alpha and their bootstrapped errors according to the 2D mode
    k, k_err, alpha, alpha_err = get2Dhist_k_alpha_err(d)
    print(filename, k, k_err, alpha, alpha_err)

# or directly apply it on the grouped dataframe to get a new dataframe
aggregated_data = data.groupby("filename").apply(get2Dhist_k_alpha_err)
print(aggregated_data)
