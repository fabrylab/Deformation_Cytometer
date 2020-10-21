# -*- coding: utf-8 -*-
"""
This programs reads multiple experients, performs a fit on the data from all experiments at once and
plots the results (stress-strain curve) in one figure. For each pressure a new curve is plotted.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from deformationcytometer.evaluation.helper_functions import fitStiffness
from deformationcytometer.evaluation.helper_functions import initPlotSettings, plotStressStrain, plotMessurementStatus
from deformationcytometer.evaluation.helper_functions import storeEvaluationResults, load_all_data, get_folders

""" loading data """
# get the results file (by config parameter or user input dialog)

# full path of the output file
out_put_file = r"/home/user/Desktop/out.pdf"

# list of base folders to analyze. Each folder must contain subfolders "1", "2" .. for different experiments.
base_folders= [r"/home/user/Desktop/biophysDS/emirzahossein/microfluidic cell rhemeter data/microscope4/2020_july/2020_07_29_aslginate2%_NIH_diff_x_position_2/inlet/",
               #r"\\131.188.117.96\biophysDS\emirzahossein\microfluidic cell rhemeter data\microscope4\2020_july\2020_07_29_aslginate2%_NIH_diff_x_position_3\inlet\\"
               ]
base_folders_ = [os.path.join(b, os.path.join("[0-9]","*_result.txt")) for b in base_folders]
print(base_folders_)
# reading all data, with out splitting for the pressure
data_all, config_all = load_all_data(base_folders_, pressure=None)
# reading the data for each pressure individually. This is needed to later plot stress-strain curves
# for each pressure separately
data1, config1 = load_all_data(base_folders_, pressure=1)
data2, config2 = load_all_data(base_folders_, pressure=2)
data3, config3 = load_all_data(base_folders_, pressure=3)



""" evaluating data"""
# fit with all pressures combined
p = fitStiffness(data_all, config_all)
config1["fit"] =  config_all["fit"]
config2["fit"] =  config_all["fit"]
config3["fit"] =  config_all["fit"]


""" plotting data """

initPlotSettings()

# add multipage plotting
pp = PdfPages(out_put_file)

# generate the velocity profile plot
#plotVelocityProfile(data, config)
#pp.savefig()
#plt.cla()

# generate the stress strain plot
# each pressure is plotted separately
plotStressStrain(data1, config1, color="C0", mew=1.5)
plotStressStrain(data2, config2, color="C1", mew=1.5)
plotStressStrain(data3, config3, color="C2", mew=1.5)
plt.gca().plot(0, 0, color="C0", label="1 bar")
plt.gca().plot(0, 0, color="C1", label="2 bar")
plt.gca().plot(0, 0, color="C2", label="3 bar")
plt.legend()
pp.savefig()

# generate the info page with the data
plotMessurementStatus(data_all, config_all)

pp.savefig()
#plt.show()
pp.close()

# store the evaluation data in a file
#storeEvaluationResults(data_all, config_all)