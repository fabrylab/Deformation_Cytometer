import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data_new, get2Dhist_k_alpha_err

# load all the data in the given folder
# you can also add a list of folders instead
# filenames can also include * for wildcard matches
data, config = load_all_data_new(r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope_1\july_2021\2021_07_05_desmin_cytoD")

# print all the columns, these are the ones from the results.csv and from the meta data files
print(data.columns)

# you can now group by some of the columns
# for example just by the filename to get an evaluation per file
for filename, d in data.groupby("filename"):
    # calculate the k and alpha and their bootstrapped errors according to the 2D mode
    k, k_err, alpha, alpha_err = get2Dhist_k_alpha_err(d)
    print(filename, k, k_err, alpha, alpha_err)

# or directly apply it on the grouped dataframe to get a new dataframe
aggregated_data = data.groupby("filename").apply(get2Dhist_k_alpha_err)
print(aggregated_data)
