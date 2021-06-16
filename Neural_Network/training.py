# This script is used to train a U-net neural Network to detect cells in the flow cytometer channel.
# Refer to .. Supplementary information section
# "Adaptation of the neural network for other cell types and experimental conditions." for deailed instructions.
# Author Andreas Bauer
# Contact: Andreas.b.Bauer@fau.de

from Neural_Network.includes.training_functions import *
from Neural_Network.includes.loss_functions import *
from Neural_Network.includes.data_handling import *
from Neural_Network.includes.weight_maps import extract_edge, weight_background
from deformationcytometer.detection.includes.UNETmodel import UNet
from tensorflow_addons.optimizers import RectifiedAdam

# list of paths to ClickPoints databases  that contain the training data
search_path1 = '/home/johannes/data2/jbartl/BasicDataSet_10-6-21/train/Immune_cells/'
cdb_files1 = find_cdb_files(search_path1)


""" the following parameters need to be set according to your data """

# number of images taken from databases in cdb_files1 and cdb_files2; use "all" for all images
n_images1 = 50
#n_images2 = 50
# set whether to use all images or only images with marked cells for training
empty_images1 = False
#empty_images2 = False
# joining everything to a list; you can add as many groups as desired
cdb_files_list = [[cdb_files1, empty_images1, n_images1]]#, [cdb_files2, empty_images2, n_images2]]
# Shape of the input images.The program will try to transpose images if the shape of the images is
# equivalent to the transposed # im_shape. You can set im_shape= None if your are sure,
# #that all images have the same size
im_shape = (540, 720)
# Full path to a weight file if you are performing Transfer Learning (which is highly recommended)
weight_path = None
# Path where logs, networks and predictions during training are stored.
output_path = "./"
# Name of the log folder and part of the network filename.
log_name = "network_training"


"""  The following parameters can be changed to improve the network training """

# If True the training data is loaded from the disk during training. Else all data is stored in memory.
# 200 images with weights use up 2 GB of memory.
train_from_disc = True
# Extracting the training data from ClickPoints files or using existing training data. This needs to be False
# if you use training_from_disc for the first time and if you change any aspect of the data generation.
use_existing_data = False
# Location where the training data is stored and/or loaded from. The directory will be created automatically
# if it doesnt already yet. The new directory will contain three subdirectories (x_data, y_data, w_data)
# If the subdirectories already existed they will be completely emptied before new data is written.
dir_training_data = "./"

# Additional training settings
# number of training epochs. Usually training shows no more progress after 40 epochs
epochs = 40
# learning rate of the training. We don't recommend to change this.
learning_rate = 1e-3
# increasing the training batch size can speed up convergence
batch_size = 4
# Size of the training test split.
test_size = 0.2
# set whether the best network weights (according to the metric function) or the last weights are saved during training
save_best_only = False

# TODO: reformualte this part???
# weighting settings
# background weight. A small background weight is typically necessary when training
# the network to only recognize the cell edge
w_bg = 0.1
# weighting is defined by a dictionary specifying the weighting function (key: "function") and any number of
# additional keyword arguments that are passed to the weighting function (in this case "ws", which is a tuple specifying
# the weight of the fore- and background pixels).
# Concerning weights: The weighting function's first argument must be a 2 dimensional array. Weights are added as
# 2-dimensional maps to the last axis of the ground truth data. The resulting array has the shape [batch_size, x, y, 2].
# loss functions must be able to unpack the ground truth and training data accordingly.
# You can use weighting = None to avoid adding a weight map all together or set "ws":[1,1] to keep the weight map
# without actually weighting anything.
weighting = {"function": weight_background, "ws": (1, w_bg)}
# loss function. This function must be able to unpack the weighted input. We had great success with binary focal loss
loss = normal_binary_focal_loss_weighted_input(gamma=2.0)
# This function is called after each epoch and evaluates the training progress on a subset of the Data (size defined in
# test_size)
# this function must be able to unpack the ground truth and weights if weights are used.
metric = accuracy_weighted_input
# string of the function name
metric_name = metric.__name__


# Random see for reproducibility. You should also load a fixed weight file to the U-net (parameter "weight_path")!
seed = 100

# Function that manipulates the ground truth
# First argument must be a 2-dimensional integer array, must also return a 2-dimensional integer array
# Our strategy is to train the network only to recognize the edge (3 pixel thickness) of cells.
# !!! Note that this approach is also relevant in the downstream cell detection process
# (e.g. in the "mask_to_cells_edge" function) and cannot simply be changed here.
mask_function = extract_edge


# Constructing the Neural Network and loading weight files
np.random.seed(seed)
unet = UNet((im_shape[0], im_shape[1], 1), 1, d=8, weights=weight_path)

# defining paths to write training data
dir_x = os.path.join(dir_training_data, "X_data")
dir_y = os.path.join(dir_training_data, "y_data")
dir_w = os.path.join(dir_training_data, "w_data")

# loading training data from ClickPoints databases and optionally writing it to the disk.
if not use_existing_data:
    write_training_data(cdb_files_list, dir_x, dir_y, seed, test_size, dir_w=dir_w, final_shape=im_shape,
                        weighting=weighting, mask_function=extract_edge)
if train_from_disc:
    # setting up data generators for training
    gen = setup_generators(dir_x=dir_x, dir_y=dir_y, dir_w=dir_w, batch_size=batch_size, seed=seed,
                           target_size=im_shape, training=True)
    len_x_data = x_data_lenght(dir_x)
    val_data = read_val_data(dir_x, dir_y, dir_w)
else:
    X_train, X_test, y_train, y_test= load_data_from_database(cdb_files_list, seed, test_size,
                                final_shape=im_shape, weighting=weighting, mask_function=extract_edge)
    # setting up data generators for training
    gen = setup_generators(X=X_train, y=y_train, batch_size=batch_size, seed=seed, target_size=im_shape)
    len_x_data = len(X_train)
    val_data = X_test, y_test

# setting up callbacks:
# after each training epoch, the new network performs one prediction on an image from the validation data set. The
# probability map of the prediction is stored in a sub folder of output_path.
X_test, y_test = next(gen)  # also find the length
callbacks = setup_callbacks_and_logs(output_path, log_name, test_size, seed, X_test, y_test, metric_name=metric_name,
                                     save_best_only=False)

# network compilation and training
unet.compile(optimizer=RectifiedAdam(lr=learning_rate), loss=loss, metrics=[metric])
history = unet.fit_generator(gen,
                             validation_data=val_data,
                             epochs=epochs,
                             steps_per_epoch=len_x_data // batch_size,
                             callbacks=callbacks)

