from sklearn.model_selection import train_test_split
from natsort import natsorted
import clickpoints
from PIL import Image
import os
import numpy as np
from matplotlib.path import Path
import shutil
from collections import defaultdict
from Neural_Network.weight_maps import generate_weight_map

# conversion to uint8 and int 32 when storing training data on the disc
uint8_max = 255
int32_max = 2147483647


def normalize_uint8(im):
    return im / uint8_max


def normalize_int32(im):
    return im / int32_max


def convert_unit8(im):
    # array values should be in (0,1)
    im_max = im.max()
    if im_max != 0:
        im = (im / im.max()) * uint8_max
    return im.astype(np.uint8)


def convert_int32(im):
    # array values should be in (0,1)
    im_max = im.max()
    if im_max != 0:
        im = (im / im.max()) * int32_max
    return im.astype(np.int32)


# preprocessing by normalizing each image indvidually
def preprocess(img):
    return (img - np.mean(img)) / np.std(img)


def preprocess_batch(x):
    return (x - np.mean(x, axis=(1, 2))[:, None, None, :]) / np.std(x, axis=(1, 2))[:, None, None, :]


def get_image_list(cdb_files, all_images, n):
    # generating a list of all images, so that we can apply random filtering

    # read all image ids from all databases
    img_id_list = []
    for cdb_file in cdb_files:
        cdb = clickpoints.DataFile(cdb_file)
        q_poly = cdb.getPolygons()
        if all_images:  # use all images
            img_ids = np.unique([im.id for im in cdb.getImages()])
        else:  # Train just on image with polygons
            img_ids = np.unique([pol.image.id for pol in q_poly])
        img_id_list.extend([(id, cdb_file) for id in img_ids])
        cdb.db.close()

    # randomly selecting n images from the databases
    n = len(img_id_list) if n == "all" else n
    img_id_list = [img_id_list[ind] for ind in np.random.choice(np.arange(len(img_id_list), dtype=int), n)]
    # splitting up to dictionary db_path: [id_list]
    img_dict = defaultdict(list)
    for im_id, db_file in img_id_list:
        img_dict[db_file].append(im_id)

    return img_dict


def read_img_mask(cdb, id, ny, nx, points, final_shape=None):
    # extracting a single image and the ground truth mask form a ClickPoints Database
    # This function tries to transpose the image to achieve image dimension set by final_shape

    if final_shape is None:
        final_shape = (ny, nx)

    mask = np.zeros((ny, nx), dtype=np.uint8)
    img_o = cdb.getImage(id=id)
    img = img_o.get_data()
    # get polygons in this image
    q_polys = cdb.getPolygons(image=img_o)
    polys = np.array([[e.points] for e in q_polys])
    # iterate through all cells, fill the polygons and add the cell area to the mask
    for pol in q_polys:
        if np.shape(pol)[0] != 0:
            polygon = np.array([[pol.points]])
            if np.sum(polygon.shape) > 7:  # single point polygon can happen on accident when clicking
                path = Path(polygon.squeeze())
                grid = path.contains_points(points)
                grid = grid.reshape((ny, nx))
                mask += grid
        if len(img.shape) == 3:
            img = img[:, :, 0]  # ToDo add real RGB conversion
    # try transposition to achieve image dimension set of final_shape
    if final_shape == mask.shape[::-1]:
        img = img.T
        mask = mask.T
    return img, mask


def grab_image_from_database(cdb_file, img_ids, final_shape=None):
    # Iterator function that extracts images and the ground truth masks
    # form a ClickPoints Database.
    # This function tries to transpose the image to achieve image dimension set by final_shape

    # opening the ClickpointDatabase
    cdb = clickpoints.DataFile(cdb_file)
    # getting the coordinates of the ground truth mask. This is needed to
    # later fill the cell area in to the mask.
    im_shape = cdb.getImages()[0].getShape()
    nx, ny = im_shape[1], im_shape[0]
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    print("using images %d" % len(img_ids))

    # iteration over all images in the database
    for id in img_ids:
        yield read_img_mask(cdb, id, ny, nx, points, final_shape=final_shape)

    # Closing the database connection after last image is read
    cdb.db.close()


def make_and_clean_folders(dir_x, dir_y, dir_w):
    # Setting up and cleaning the folder where the training data is stored

    for d in [dir_x, dir_y, dir_w]:
        if not d is None:
            if os.path.exists(d):
                shutil.rmtree(d)
                os.makedirs(d, exist_ok=True)
            else:
                os.makedirs(d, exist_ok=True)


def split_dir_training_test(dir, test_size, random_state):
    # train validation split of training data when training from disk

    np.random.seed(random_state)
    # finding all files
    files = natsorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".tif")])
    files = np.array(files)
    # spliting in train and test data
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)
    # moving train data
    target_train = os.path.join(dir, "train", "class1")
    os.makedirs(target_train, exist_ok=True)
    for f in train_files:
        f_ = os.path.split(f)[1]
        shutil.move(f, os.path.join(target_train, f_))
    # moving the validation data
    target_test = os.path.join(dir, "test", "class1")
    os.makedirs(target_test, exist_ok=True)
    for f in test_files:
        f_ = os.path.split(f)[1]
        shutil.move(f, os.path.join(target_test, f_))


def read_val_data(dir_x, dir_y, dir_w=None):
    # reading the validation data set from the disk. The entire validation data set is read in the memory
    # TODO can I imporve tis

    # Reading and preprocessing validation images
    x_path = os.path.join(dir_x, "test", "class1")
    files_x = [os.path.join(x_path, im) for im in os.listdir(x_path) if im.endswith(".tif")]
    x = np.array([np.array(Image.open(im)) for im in files_x])[:, :, :, None]
    x = preprocess_batch(x)
    # reading the validation ground truth
    y_path = os.path.join(dir_y, "test", "class1")
    files_y = [os.path.join(y_path, im) for im in os.listdir(y_path) if im.endswith(".tif")]
    y = []
    for im in files_y:
        im_ = Image.open(im)
        if im_.mode == "L":
            y.append(normalize_uint8(np.array(im_)))
        else:
            raise Exception("Failed to load mask. Excpected image mode L (uint8) for mask data")
    y = np.array(y)[:, :, :, None]

    # optionally adding weight data
    if not dir_w is None:
        w_path = os.path.join(dir_w, "test", "class1")
        files_w = [os.path.join(w_path, im) for im in os.listdir(w_path) if im.endswith(".tif")]
        w = []
        for im in files_w:
            im_ = Image.open(im)
            if im_.mode == "I":
                w.append(normalize_int32(np.array(im_)))
            else:
                raise Exception("Failed to load mask. Excpected image mode I (int32) for mask data")
        w = np.array(w)[:, :, :, None]
        y = np.concatenate([y, w], axis=3)
    print("loaded val data / shape: ", x.shape, y.shape)

    return x, y


def write_training_data(cdb_list, dir_x, dir_y, random_state, test_size, dir_w=None, final_shape=None, weighting=None,
                        mask_function=None):
    # extracting images and ground truth data from ClickPoints databases, generating weights and writing them to the disk

    counter = 0
    np.random.seed(random_state)
    # setting up folders
    make_and_clean_folders(dir_x, dir_y, dir_w)
    # iterating through database groups
    for (cdb_files, all_images, n) in cdb_list:
        # randomly selecting n images from the database group
        img_dict = get_image_list(cdb_files, all_images, n)
        # iterating through databases in one group
        for cdb_file, img_ids in img_dict.items():
            # iterating through individual images
            for img, mask in grab_image_from_database(cdb_file, img_ids, final_shape=final_shape):
                # tensorflow from directory on reads type "L","I" and "I:16", (uint 8 and signed integers 32 and 16)
                # --> see load_data in /home/user/anaconda3/lib/python3.7/site-packages/keras_preprocessing/image/utils.py
                Image.fromarray(img, mode="L").save(os.path.join(dir_x, str(counter) + ".tif"))
                # Manipulate the ground truth mask. We typically reduce the mask to only cover the cell edge.
                if not mask_function is None:
                    mask = mask_function(mask)
                # saving the mask as tif
                mask = convert_unit8(mask)
                Image.fromarray(mask, mode="L").save(os.path.join(dir_y, str(counter) + ".tif"))
                # generating and saving weights
                if not weighting is None and not dir_w is None:
                    weight = generate_weight_map(mask, **weighting)
                    weight = convert_int32(weight)  # conversion to int 32 scale
                    Image.fromarray(weight, mode="I").save(os.path.join(dir_w, str(counter) + ".tif"))
                counter += 1

        # splitting into train and test data
        split_dir_training_test(dir_x, test_size, random_state)
        split_dir_training_test(dir_y, test_size, random_state)
        if not dir_w is None:
            split_dir_training_test(dir_w, test_size, random_state)


def load_data_from_database(cdb_list, random_state, test_size, final_shape=None, weighting=None, mask_function=None):
    # extracting images and ground truth data from ClickPoints databases, generating weights and storing everything
    # in the memory

    imgs = []
    masks = []
    weights = []
    # iterating through database groups
    for (cdb_files, all_images, n) in cdb_list:
        # randomly selecting n images from the database group
        img_dict = get_image_list(cdb_files, all_images, n)
        # iterating through databases in one group
        for cdb_file, img_ids in img_dict.items():
            # iterating through individual images
            for img, mask in grab_image_from_database(cdb_file, img_ids, final_shape=final_shape):
                # Manipulate the ground truth mask. We typically reduce the mask to only cover the cell edge.
                if not mask_function is None:
                    mask = mask_function(mask)
                imgs.append(img)
                masks.append(mask)
                # saving weights
                if not weighting is None:
                    weight = generate_weight_map(mask, **weighting)
                    weights.append(weight)
    # adding a dimension for the channel to the training data
    # we use this channel to store the weights in theground truth data array
    imgs = np.array(imgs)[:, :, :, None]
    masks = np.array(masks)[:, :, :, None]
    weights = np.array(weights)[:, :, :, None]
    # splitting into train and test data
    X_train, X_test = train_test_split(imgs, test_size=test_size, random_state=random_state)
    y_train, y_test = train_test_split(masks, test_size=test_size, random_state=random_state)
    w_train, w_test = train_test_split(weights, test_size=test_size, random_state=random_state)
    # adding weights
    if not weighting is None:
        y_train = np.concatenate([y_train, w_train], axis=3)
        y_test = np.concatenate([y_test, w_test], axis=3)

    print("training data shape:", np.shape(X_train), "test data shape:", np.shape(X_test))

    return X_train, X_test, y_train, y_test

def find_cdb_files(search_path):
    file_list = []
    for file in os.listdir(search_path):
        if not file.endswith(".cdb") or os.path.isfile(file) or "evaluated" in file or "test" in file:
            continue
        file_list.append(os.path.join(search_path, file))

    return file_list


def x_data_lenght(x_dir):
    x_dir = os.path.join(x_dir, "train", "class1")
    files = [f for f in os.listdir(x_dir) if ".tif" in f]
    return len(files)




