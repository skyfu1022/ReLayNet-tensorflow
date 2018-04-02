import tensorflow as tf
from scipy.io import savemat
from tensorflow.contrib import slim
import numpy as np

def cal_weight(label, contour, w1, w2):
    contour[contour != 0] = 1
    label = label.copy()
    label[label == 8] = 0
    label[label != 0] = 1
    weight = 1 + w1 * contour + w2 * label
    # savemat('test.mat', {'label': label, 'weight': weight})
    return weight

def dice_loss(probs, labels, epsilon=0.0001, scope='dice_loss'):
    with tf.variable_scope(scope):
        # probs = tf.nn.softmax(logits)
        nume = tf.reduce_sum(tf.multiply(probs, labels), axis=[1, 2]) + epsilon
        deno = tf.reduce_sum(probs ** 2 + labels ** 2, axis=[1, 2]) + epsilon
        dice = 1 - tf.reduce_mean(2 * nume / deno)
        return dice


def mce_loss(probs, labels, weights, epsilon=0.0001, scope='dice_loss'):
    with tf.variable_scope(scope):
        mce = tf.reduce_mean(tf.reduce_sum(- labels * tf.log(probs + epsilon), axis=-1, keep_dims=True) * weights)
    return mce
	

def f1_score_metrix(one_hot_label, one_hot_pred, num_classes, scope='f1_score'):
    with tf.variable_scope(scope):
        epsilon = tf.constant(value=1e-6)
        flat_label = tf.reshape(tf.cast(tf.reshape(one_hot_label, [-1, num_classes]), tf.uint8), [-1, num_classes])
        flat_pred = tf.reshape(tf.cast(tf.reshape(one_hot_pred, [-1, num_classes]), tf.uint8), [-1, num_classes])
        TP = tf.to_float(tf.count_nonzero(flat_pred * flat_label, axis=0))
        TN = tf.to_float(tf.count_nonzero((flat_pred - 1) * (flat_label - 1), axis=0))
        FP = tf.to_float(tf.count_nonzero(flat_pred * (flat_label - 1), axis=0))
        FN = tf.to_float(tf.count_nonzero((flat_pred - 1) * flat_label, axis=0))
        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        f1_score = 2 * precision * recall / (precision + recall + epsilon)
        return f1_score


def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret