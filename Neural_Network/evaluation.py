from Neural_Network.includes.network_evaluation_functions import *
from Neural_Network.includes.network_evaluation_display import *

#### Evaluation parameters #######

# output folder for data and plots
out_folder = '.'
# list of .h5 network weight files. Provide the full path to each file.
# You can add our classical segmentation algorithm by using "classical" as one element of the network_list
network_list = [r"/home/user/network1.h5",
                r"/home/user/network2.h5"]

# list of .cdb ground truth databases. Provide the full path to each file.
# These databases should not have been used for training at all.
gt_list = [r"/home/user/gt1.cdb",
           r"/home/user/gt3.cdb"]


##### Additional parameters #####

# default imaging and filter parameters
# r_min (in µm) and edge_dist (in µm) are important filters defining
# the minimal size and the minimal distance of a cell to the horizontal image edges
# the final pixel size os calculated by pixel_size_camera / (magnification * coupler)
para_dict = {"magnification": 40,
             "coupler": 0.5, "pixel_size_camera": 6.9, "r_min": 2, "edge_dist": 15,
             "channel_width": 0}

# performing a new evaluation. If False the script will only read an existing result.pickle file in out_folder and
# generate plots from it.
new_analysis = True
# threshold for solidity
sol_th = 0.96
# threshold for regularity
irr_th = 1.06
# apply post processing steps for network that detects only the cell edge
# (all cells without fully closed edge are discarded
edge_only = [True] * len(network_list)
# batch size for neural network evaluation. Larger batch sizes increase the evaluation speed but
# may not be supported by your CPU/GPU
batch_size = 5
# separately plot the results without applying regularity and solidity filters.
# r_min and edge_dist filters are applied in any case
plot_filtered = True
# Dictionary that assigns each path to a network weight file a name that is displayed when plotting.
# the default simply uses the filename of the weight file (without extension).
network_names = {n: os.path.splitext(os.path.split(n)[1])[0] for n in network_list}
# pylustrator is a useful tool to produce publication ready figures
use_pylustrator = True

# evaluation
if new_analysis:
    os.makedirs(out_folder, exist_ok=True)
    # Setting up empty data frames that are filled at each iteration of the evaluation function
    result = pd.DataFrame({"database": "", "network": "", "network_path": "", "GT": [], "Pred": [],
                           "GT Match": [], "Pred Match": [], "Pred empty frame": [], "regularity": []})
    result_filtered = result.copy()
    # looping over all weight files and ground truth databases
    for cdb_file in gt_list:
        for eo, network in zip(edge_only, network_list):
            print("database_file", cdb_file, "network", network)
            # evaluating a network on a ground truth database
            evaluate_database(result, result_filtered, cdb_file, network, para_dict,
                              network_names, edge_only=eo, irr_th=irr_th, sol_th=sol_th, batch_size=batch_size)

    # adding measures for error between ground truth and prediction (error of strain, recall , precision etc...)
    add_measures(result)
    add_measures(result_filtered)
    # adding total number of detection
    result["total detections"] = [len(i) for i in result["Pred"]]
    result_filtered["total detections"] = [len(i) for i in result_filtered["Pred"]]
    # writing results with full list of matches
    result.to_pickle(os.path.join(out_folder, "result_data_frame.pickle"))
    result_filtered.to_pickle(os.path.join(out_folder, "result_data_frame_filtered.pickle"))
    # shorter version that can be opened in excel
    write_result_summary(result, out_folder, "evaluation_results.csv")
    write_result_summary(result_filtered, out_folder, "evaluation_results_filtered.csv")

# opening the result files
result = pd.read_pickle(os.path.join(out_folder, "result_data_frame.pickle"))
result_filtered = pd.read_pickle(os.path.join(out_folder, "result_data_frame_filtered.pickle"))

# display of the evaluation results
width = 0.2
dist1 = width * 1.2
if use_pylustrator:
    import pylustrator

    pylustrator.start()
    fig1 = plot_results(result, out_folder, "network_evaluation.png", width=width, dist=dist1,
                        use_pylustrator=use_pylustrator)
    plt.show()
    if plot_filtered:
        pylustrator.start()
        fig2 = plot_results(result_filtered, out_folder, "network_evaluation_filtered.png", width=width,
                            dist=dist1, use_pylustrator=use_pylustrator)
        plt.show()
else:
    fig1 = plot_results(result, out_folder, "network_evaluation.png", width=width, dist=dist1,
                        use_pylustrator=use_pylustrator)
    if plot_filtered:
        fig2 = plot_results(result_filtered, out_folder, "network_evaluation_filtered.png", width=width,
                            dist=dist1, use_pylustrator=use_pylustrator)
