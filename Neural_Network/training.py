# publication ready scrip for network training

from Neural_Network.training_functions import *
from Neural_Network.loss_functions import *
from Neural_Network.data_handling import *
from Neural_Network.weight_maps import extract_edge, weight_background
from deformationcytometer.detection.includes.UNETmodel import UNet
from tensorflow_addons.optimizers import RectifiedAdam

# list of paths to ClickPoints databases  that contain the training data
search_path1 = '/home/user/Desktop/2020_Deformation_Cytometer/data/train/outline_GT_old_setup'
search_path2 = '/home/user/Desktop/2020_Deformation_Cytometer/data/train/gt_celltypes'
cdb_files1 = find_cdb_files(search_path1)
cdb_files2 = find_cdb_files(search_path2)



#######     the following parameters need to be set according to your data #########
# You can set the number of images and whether to use all images or only images with marked cells for each
# databases separately
# number of images taken from databases in cdb_files1; use "all" for all images
n_files1 = 50
n_files2 = 50 # number of images taken from databases in cdb_files2; use "all" for all images
# set whether to use all images or only images with marked cells for training
empty_images1 = False
empty_images2 = False
# joining everything to a list; you can add as many groups as desired
cdb_files_list = [[cdb_files1, empty_images1, n_files1], [cdb_files2, empty_images2, n_files2]]
# Shape of the input images.The program will try to transpose images if the shape of the images is equivalent to the transposed
# im_shape. You can set im_shape= None if your are sure, that all images have the same size
im_shape = (540, 720)
# Full path to a weight file if you are performing Transfer Learning (which is highly recommended)
weight_path = None
# Writing the training data to the disk. This needs to be True, if you use training_from_disc for the first time and if
# you change any aspect of the data generation.
write_data = False
# If True the training data is loaded from the disk during training. Else all data is stored in memory.
# 200 images with weights use up 2 GB of memory.
# # TODO: the Memory usage seems excessive ??
train_from_disc = False
# Location where the tranining data is stored if write_data is True. Directory will be created if it doesnt exist yet.
# Additionally 3 new directories (x_data, y_data, w_data) will be created. These subdirectories are completely
# emptied before new data is written
dir_traing_data = "."
# Path where logs, networks and predictions during training are stored.
output_path = "."
# Name of the log folder and part of the network filename.
log_name = "network_training"



#######     the following parameters need to can be changed to imporve the network training #########
# Traing settings
epochs = 40 # number of training epochs. Usually training shows no more progress after 40 epochs
learning_rate = 1e-3 # learning rate of the training. We don't recommand to change this.
batch_size = 4 # increasing the training batch size can speed up convergence
# Size of the training test split.
test_size = 0.2
# Set whether the best network (according to the metric function) or the newest network version is saved during traing
save_best_only =  False

# TODO: reformualte this part???
# weighting settings
w_bg = 0.1 # small background weight is likely to be necessary when training othe nework to recognize only the cell edge
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


# Random see for reproduceabillity. You should also load a fixed weight file to the unet (parameter weight_path)!
seed = 100

# Function that manipulates the ground truth
# First argument must be a 2-dimensional array.
# Our strategy is to train the network only to recognize the edge (3 pixel thickness) of cells.
# !!! Note that this approach is also relevant in the downstream cell detection process (e.g. in the "mask_to_cells_edge" function)
# and cannot simply be changed here.
mask_function = extract_edge


# Constructing the Neural Network and loading weight files
np.random.seed(seed)  # 121
unet = UNet((im_shape[0], im_shape[1], 1), 1, d=8, weights=None)
if not weight_path is None:
    unet.load_weights(weight_path)

# defining paths to write training data
dir_x = os.path.join(dir_traing_data, "X_data")
dir_y = os.path.join(dir_traing_data, "y_data")
dir_w = os.path.join(dir_traing_data, "w_data")

# loading training data from clickpoints databases and optionally writing it to the disk.
# setting up data generators for training
if write_data:
    write_training_data(cdb_files_list, dir_x, dir_y, seed, test_size, dir_w=dir_w, final_shape=im_shape,
                        weighting=weighting, mask_function=extract_edge)
if train_from_disc:
    gen = setup_generators(dir_x=dir_x, dir_y=dir_y, dir_w=dir_w, batch_size=batch_size, seed=seed,
                           target_size=im_shape, training=True)
    len_x_data = x_data_lenght(dir_x)
    val_data = read_val_data(dir_x, dir_y, dir_w)
else:
    X_train, X_test, y_train, y_test= load_data_from_database(cdb_files_list, seed, test_size, final_shape=im_shape, weighting=weighting, mask_function=extract_edge)
    gen = setup_generators(X=X_train, y=y_train, batch_size=batch_size, seed=seed, target_size=im_shape)
    len_x_data = len(X_train)
    val_data = X_test, y_test

# setting up callbacks:
# after each training epoch, the new network performs one prediction on an image from the validation data set. The
# probability map of the prediction is stored in a sub folder of output_path
X_test, y_test = next(gen)  # also find the length
callbacks = setup_callbacks_and_logs(output_path, log_name, test_size, seed, X_test, y_test, metric_name=metric_name,
                                     save_best_only=False)

# network compilation and training
unet.compile(optimizer=RectifiedAdam(lr=learning_rate), loss=loss, metrics=metric)
history = unet.fit_generator(gen,
                             validation_data=val_data,
                             epochs=epochs,
                             steps_per_epoch=len_x_data // batch_size,
                             callbacks=callbacks)

