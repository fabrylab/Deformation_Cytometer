
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import imageio
import numpy as np
from Neural_Network.data_handling import preprocess_batch, normalize_uint8, normalize_int32



class SinglePredictionImage(Callback):
    def __init__(self, image, save_path, label=None, split_shape=None, verbose=0, batch_size=None, normalize=False):
        """
        :param image:
        the single image to predict on
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(SinglePredictionImage, self).__init__()

        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.savepath = save_path
        self.split_shape = split_shape
        self.image = image
        self.normalize = normalize
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        imageio.imwrite(self.savepath.format(epoch="input"), (((image[:,:,0] - image.min())/(image.max() - image.min())) * 255).astype(np.uint8))
        if label is not None:
            imageio.imwrite(self.savepath.format(epoch="mask"), (label * 255).squeeze().astype(np.uint8))

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)

        if self.split_shape is not None:
            out = np.array([self.model.predict(img[None, :])[0] for img in self.image])
            out = np.hstack(
                np.hstack(out.reshape((self.split_shape[0], self.split_shape[1], *self.image[0].shape[:2]))))
        else:
            out = self.model.predict(self.image[None, :])

        if self.normalize:
            out = (out - np.amin(out)) / (np.amax(out) - np.amin(out))
        imageio.imwrite(self.savepath.format(epoch=epoch), (out * 255).squeeze().astype(np.uint8))


## check if tf detected a GPU
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x for x in local_device_protos if x.device_type == 'GPU']


# wrapper for parallel augmentations, generator for image masks combos
def data_gen_comb_from_dir(gen_img, gen_mask, dir_x, dir_y, dir_w = None, batch_size = 128, target_size = None, seed = np.random.randint(100)):

    weights_included =  not dir_w is None # when y contains a weightmap
    genx = gen_img.flow_from_directory(dir_x, target_size=target_size, color_mode="grayscale",
                                       class_mode="binary", batch_size=batch_size, seed=seed)
    geny = gen_mask.flow_from_directory(dir_y, target_size=target_size, color_mode="grayscale",
                                        class_mode="binary", batch_size=batch_size, seed=seed)
    if weights_included:
            genw = gen_mask.flow_from_directory(dir_w, target_size=target_size, color_mode="grayscale",
                                                 class_mode="binary", batch_size=batch_size, seed=seed)
    while True:
        Xi, _ = genx.next()
        yi, _ = geny.next()
        Xi = preprocess_batch(Xi)
        yi = normalize_uint8(yi)
        if not weights_included:
            wi, _ = genw.next()
            wi = normalize_int32(wi) # converts int32 image to floats in range 0,1
            yi = np.concatenate([yi.astype(np.float32), wi.astype(np.float32)], axis=-1)
        yield Xi, yi

def datagen_comb(gen_img, gen_mask, X, y, batch_size=128, target_size=None, seed=np.random.randint(100)):
    # works for both X and
    # just make sure the seed is identical - so modifications are applied identical on both samples
    weights_included =  y.shape[-1] == 2 # when y contains a weightmap

    genx = gen_img.flow(X, np.ones(len(X)), seed=seed, batch_size=batch_size)
    if weights_included:
        geny = gen_mask.flow(y[:,:,:,:-1], np.ones(len(y)), seed=seed, batch_size=batch_size)
        genw = gen_mask.flow(y[:,:,:,1:], np.ones(len(y)), seed=seed, batch_size=batch_size)
    else:
        geny =  gen_mask.flow(y, np.ones(len(y)), seed=seed, batch_size=batch_size)

    while True:
        Xi, _ = genx.next()
        Xi = preprocess_batch(Xi)
        yi, _ = geny.next()
        if weights_included:
            wi, _ = genw.next()
            yi = np.concatenate([yi.astype(np.float32), wi.astype(np.float32)], axis=-1)
        yield Xi, yi



def setup_generators(X=None, y=None, dir_x=None, dir_y=None, dir_w=None, batch_size=None, seed=np.random.randint(100),
                     target_size=None, training=True):
    augmentation = {"zca_whitening": False, "samplewise_center": False, "samplewise_std_normalization": False,
                    "horizontal_flip": True, "vertical_flip": True,
                     "shear_range": 0.1}
    dirs = []
    for d in [dir_x, dir_y, dir_w]:
        if not d is None and training:
            dirs.append(os.path.join(d, "train"))
        else:
            dirs.append(d)

    datagen_img = ImageDataGenerator(**augmentation)
    datagen_mask = ImageDataGenerator(**augmentation)
    if (not dir_x is None) and (not dir_y is None):
        gen = data_gen_comb_from_dir(datagen_img, datagen_mask, dirs[0],
                       dirs[1], dir_w=dirs[2], batch_size=batch_size, target_size=target_size, seed=seed)
    elif (not X is None) and (not y is None):
        gen = datagen_comb(datagen_img, datagen_mask, X, y, batch_size=batch_size, target_size=target_size, seed=seed)
    else:
        raise Exception
    return gen



def setup_callbacks_and_logs(out_put_path, log_name, test_size, random_state, X_test, y_test, metric_name="accuracy_weighted_input",save_best_only=True):
    stats = dict(name="Unet",
                 version=log_name,
                 date=datetime.now().strftime('%Y%m%d-%H%M%S'),
                 data_raw="-".join(["M1p1", "M1p2", "M1p3"]),
                 data_split="%d-%d" % (test_size * 100, random_state),
                 model_path=out_put_path,
                 model_ext="h5",
                 log_path= out_put_path,
                 img_path= out_put_path
                 )

    # create output paths
    filepath = "{model_path}{name}_{version}_{date}.{model_ext}".format(**stats)
    img_path = "{img_path}/{name}_{version}/{epoch}.jpg".format(epoch="{epoch}", **stats)
    log_dir = '{log_path}{name}_{version}_{date}'.format(**stats)
    print("Logs:")
    print(filepath)
    print(log_dir)
    print(filepath)

    tensorboard = TensorBoard(log_dir=log_dir,
                              histogram_freq=0, write_graph=True, write_images=False, profile_batch=0)

    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor=metric_name, save_best_only=save_best_only)
    # learningrate = LearningRateScheduler(scheduler) # this could be implemented as a callback
    id = 10 if X_test.shape[0] > 10 else -1
    singleprediction = SinglePredictionImage(X_test[id, :, :, :], img_path, label=y_test[id, :, :, 0])
    callbacks = [checkpoint, tensorboard, singleprediction]
    return callbacks

# not used
def scheduler(epoch):
    learning_rate = 1e-5
    if epoch < 10:
        return learning_rate
    # if epoch < 100:
    #  return 1e-1

    if epoch < 20:
        return learning_rate * 0.1
    if epoch < 40:
        return learning_rate * 0.01
    if epoch >= 40:
        return learning_rate * 0.001


