import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import epsilon
from tensorflow.python.keras.utils.losses_utils import reduce_weighted_loss, ReductionV2


def normal_binary_focal_loss_weighted_input(gamma):

    def normal_binary_focal_loss_wi(y_true, y_pred):
        # taken from https: // focal - loss.readthedocs.io / en / latest / generated / focal_loss.BinaryFocalLoss.html
        # also check https://arxiv.org/pdf/1708.02002.pdf

        # unpacking labels and weights
        target = tf.convert_to_tensor(y_true)
        target, weights = tf.split(target,num_or_size_splits=2, axis=-1)
        target = tf.dtypes.cast(target, dtype=tf.bool)
        weights = tf.keras.backend.cast(weights, tf.float32)

        output = tf.convert_to_tensor(y_pred)
        output = tf.keras.backend.cast(output, tf.float32)


        epsilon_ = tf.convert_to_tensor(epsilon(), tf.float32)
        p = output
        q = 1 - p
        # avoid zeros in p and q --> would cause problems with log(0) later
        p = tf.math.maximum(p, epsilon_)
        q = tf.math.maximum(q, epsilon_)

        # Loss for the positive examples
        pos_loss = -(q ** gamma) * tf.math.log(p)
        # Loss for the negative examples
        neg_loss = -(p ** gamma) * tf.math.log(q)
        # choose either pos_loss or neg_loss, depending on the input label
        loss = tf.where(target, pos_loss, neg_loss) * weights
        loss = reduce_weighted_loss(loss, reduction=ReductionV2.SUM_OVER_BATCH_SIZE)
        return loss
    return normal_binary_focal_loss_wi


def normal_focal_loss(gamma=2.0):

    def _normal_focal_loss(y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.float32)
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.dtypes.cast(y_true, dtype=tf.bool)
        epsilon_ = tf.convert_to_tensor(epsilon(), tf.float32)
        p = y_pred
        q = 1 - p
        # avoid zeros in p and q --> would cause problems with log(0) later
        p = tf.math.maximum(p, epsilon_)
        q = tf.math.maximum(q, epsilon_)

        # Loss for the positive examples
        pos_loss = -(q ** gamma) * tf.math.log(p)
        # Loss for the negative examples
        neg_loss = -(p ** gamma) * tf.math.log(q)
        # choose either pos_loss or neg_loss, depending on the input label
        loss = tf.where(y_true, pos_loss, neg_loss)
        loss = reduce_weighted_loss(loss, reduction=ReductionV2.SUM_OVER_BATCH_SIZE)
        return loss

    return _normal_focal_loss


def accuracy_weighted_input(y_true, y_pred):

    target = tf.convert_to_tensor(y_true)
    target, weights = tf.split(target, num_or_size_splits=2, axis=-1)
    target = tf.dtypes.cast(target, dtype=tf.bool)

    output = tf.convert_to_tensor(y_pred)
    output = tf.keras.backend.cast(output, tf.float32)
    output = output > 0.5
    return tf.keras.backend.mean(tf.keras.backend.equal(output, target))


def normal_binary_crossentropy(y_true, y_pred):
    target = tf.convert_to_tensor(y_true)
    output = tf.convert_to_tensor(y_pred)

    epsilon_ = tf.convert_to_tensor(epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon_, 1. - epsilon_)
    # Compute cross entropy from probabilities.
    bce = target * tf.math.log(output + epsilon())
    bce += (1 - target) * tf.math.log(1 - output + epsilon())
    return -bce


def DiceLoss(y_true, y_pred, smooth=10e-6):
    # this is the "generalized dice loss??
    # flatten label and prediction tensors
    # print(y_true.shape)
    # print(y_pred.shape)

    y_pred = tf.keras.backend.flatten(y_pred)
    y_pred = tf.keras.backend.cast(y_pred, tf.float32)
    y_true = tf.keras.backend.flatten(y_true)
    y_true = tf.keras.backend.cast(y_true, tf.float32)

    intersection = tf.keras.backend.sum(y_true * y_pred)
    dice = (2 * intersection + smooth) / (tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) + smooth)
    return 1 - dice


# deprecated
def scale_weights(weights):
    _max = np.max(weights)
    return [w/_max for w in weights]


# deprecated
def DiceLoss_weighted(w1=1, w2=1, w3=1, w4=1, _scale_weights=True):
    '''
       w1 weight for inside pixel
       w2 weight for edge pixel cell
       w3 weight for edge pixel background
       w4  weight for background pixel
       '''
    w1 = np.array(w1, dtype=np.float32)
    w2 = np.array(w2, dtype=np.float32)
    w3 = np.array(w3, dtype=np.float32)
    w4 = np.array(w4, dtype=np.float32)

    if _scale_weights:
        w1,w2,w3,w4 = scale_weights([w1,w2,w3,w4])


    def dice_weighted(y_true, y_pred):

        target = tf.convert_to_tensor(y_true)
        output = tf.convert_to_tensor(y_pred)
        output = tf.keras.backend.cast(output, tf.float32)

        # retrieving the original mask
        target = tf.dtypes.cast(target, dtype=tf.int8)
        labels = (target == 1) | (target == 2)  # are these "tensor operations"
        labels = tf.keras.backend.cast(labels, tf.float32)

        # generating weights tensor
        weights = tf.ones(tf.shape(target), tf.float32)

        weights = tf.where(target == 1, w1, weights)
        weights = tf.where(target == 2, w2, weights)
        weights = tf.where(target == 3, w3, weights)
        weights = tf.where(target == 0, w4, weights)

        epsilon_ = tf.convert_to_tensor(epsilon(), tf.float32)
        intersection = tf.keras.backend.sum(labels * output * weights)
        dice = (2 * intersection + epsilon_) / (tf.keras.backend.sum(labels * weights) + tf.keras.backend.sum(output * weights) + epsilon_)
        #tf.print("dice",dice)
        return 1 - dice

    return dice_weighted



