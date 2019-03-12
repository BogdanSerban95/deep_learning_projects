import tensorflow as tf
import keras.backend as K

'''Based on the kernel found at: https://www.kaggle.com/pestipeti/explanation-of-scoring-metric'''


def cast_f(x):
    return K.cast(x, K.floatx())


def cast_b(x):
    return K.cast(x, bool)


def iou_loss_core2(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def iou_loss_core(true, pred):  # this can be used as a loss if you make it negative
    intersection = true * pred
    not_true = 1 - true
    union = true + (not_true * pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())


def comp_metric(true, pred):  # any shape can go - can't be a loss function

    tresholds = [0.5 + (i * .05) for i in range(10)]

    # flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = cast_f(K.greater(pred, 0.5))

    # total white pixels - (batch,)
    true_sum = K.sum(true, axis=-1)
    pred_sum = K.sum(pred, axis=-1)

    # has mask or not per image - (batch,)
    true1 = cast_f(K.greater(true_sum, 1))
    pred1 = cast_f(K.greater(pred_sum, 1))

    # to get images that have mask in both true and pred
    true_positive_mask = cast_b(true1 * pred1)

    # separating only the possible true positives to check iou
    test_true = tf.boolean_mask(true, true_positive_mask)
    test_pred = tf.boolean_mask(pred, true_positive_mask)

    # getting iou and threshold comparisons
    iou = iou_loss_core(test_true, test_pred)
    true_positives = [cast_f(K.greater(iou, tres)) for tres in tresholds]

    # mean of thressholds for true positives and total sum
    true_positives = K.mean(K.stack(true_positives, axis=-1), axis=-1)
    true_positives = K.sum(true_positives)

    # to get images that don't have mask in both true and pred
    true_negatives = (1 - true1) * (1 - pred1)  # = 1 -true1 - pred1 + true1*pred1
    true_negatives = K.sum(true_negatives)

    return (true_positives + true_negatives) / cast_f(K.shape(true)[0])
