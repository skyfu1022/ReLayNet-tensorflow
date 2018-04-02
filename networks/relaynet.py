import tensorflow as tf
# from tensorflow.contrib import slim
from utils.ops import unpool_with_argmax
from tensorflow.contrib.layers import variance_scaling_initializer, l2_regularizer, xavier_initializer

class ReLayNet(object):
    def __init__(self, config):
        self.config = config
        self.encoders = []
        self.indexs = []

    def conv_bnorm_relu(self, inputs):
        conv = tf.layers.conv2d(inputs, filters=self.config.num_filters, kernel_size=self.config.kernel_size,
                                  padding='same', kernel_initializer=variance_scaling_initializer(),
                                  kernel_regularizer=l2_regularizer(self.config.weight_decay), name='conv')
        bnorm = tf.layers.batch_normalization(conv, name='bnorm')
        relu = tf.nn.relu(bnorm, name='relu')
        return relu

    def encoder_block(self, inputs, scope):
        with tf.variable_scope(scope):
            encoder = self.conv_bnorm_relu(inputs)
            self.encoders.append(encoder)
            max_pool, index = tf.nn.max_pool_with_argmax(encoder, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name='maxpool')
            self.indexs.append(index)
            return max_pool

    def bottom_block(self, inputs, scope):
        with tf.variable_scope(scope):
            bottom = self.conv_bnorm_relu(inputs)
            return bottom

    def decoder_block(self, inputs, scope):
        with tf.variable_scope(scope):
            unpool = unpool_with_argmax(inputs, self.indexs.pop(), name='unpool')
            concat = tf.concat((unpool, self.encoders.pop()), axis=-1)
            decoder = self.conv_bnorm_relu(concat)
            return decoder

    def classify_block(self, inputs):
        logits = tf.layers.conv2d(inputs, filters=self.config.num_classes, kernel_size=1, padding='same',
                                  kernel_initializer=variance_scaling_initializer(),
                                  kernel_regularizer=l2_regularizer(self.config.weight_decay), name='conv')
        # prob = tf.nn.softmax(logits)
        return logits

    def __call__(self, inputs):
        with tf.variable_scope('relaynet01'):
            maxpool1 = self.encoder_block(inputs, scope='encoder1')
            maxpool2 = self.encoder_block(maxpool1, scope='encoder2')
            maxpool3 = self.encoder_block(maxpool2, scope='encoder3')
            bottom = self.bottom_block(maxpool3, scope='bottom')
            decoder1 = self.decoder_block(bottom, scope='decoder1')
            decoder2 = self.decoder_block(decoder1, scope='decoder2')
            decoder3 = self.decoder_block(decoder2, scope='decoder3')
            logits = self.classify_block(decoder3)
            return logits
