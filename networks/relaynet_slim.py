import tensorflow as tf
from tensorflow.contrib import slim
from utils.ops import unpool_with_argmax


class ReLayNet(object):
    def __init__(self, config):
        self.config = config

    def conv_bnorm_relu(self, inputs):
        conv = slim.conv2d(inputs, self.config.num_filters, self.config.kernel_size, normalizer_fn=slim.batch_norm,
                           padding='SAME', weights_initializer=slim.variance_scaling_initializer(),
                           weights_regularizer=slim.l2_regularizer(self.config.weight_decay), scope='conv')
        return conv

    def encoder_block(self, inputs, scope):
        with tf.variable_scope(scope):
            encoder = self.conv_bnorm_relu(inputs)
            max_pool, index = tf.nn.max_pool_with_argmax(encoder, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name='maxpool')
            return max_pool, index, encoder

    def bottom_block(self, inputs, scope):
        with tf.variable_scope(scope):
            bottom = self.conv_bnorm_relu(inputs)
            return bottom

    def decoder_block(self, inputs, decoder, index, scope):
        with tf.variable_scope(scope):
            unpool = unpool_with_argmax(inputs, index, name='unpool')
            concat = tf.concat((unpool, decoder), axis=-1)
            decoder = self.conv_bnorm_relu(concat)
            return decoder

    def classify_block(self, inputs):
        logits = slim.conv2d(inputs, self.config.num_classes, 1, padding='SAME', activation_fn=None,
                             weights_initializer=slim.variance_scaling_initializer(),
                             weights_regularizer=slim.l2_regularizer(self.config.weight_decay), scope='conv')
        # prob = tf.nn.softmax(logits)
        return logits

    def __call__(self, inputs):
        with tf.variable_scope('relaynet01'):
            maxpool1, index1, encoder1 = self.encoder_block(inputs, scope='encoder1')
            maxpool2, index2, encoder2 = self.encoder_block(maxpool1, scope='encoder2')
            maxpool3, index3, encoder3 = self.encoder_block(maxpool2, scope='encoder3')
            bottom = self.bottom_block(maxpool3, scope='bottom')
            decoder1 = self.decoder_block(bottom, encoder3, index3, scope='decoder1')
            decoder2 = self.decoder_block(decoder1, encoder2, index2, scope='decoder2')
            decoder3 = self.decoder_block(decoder2, encoder1, index1, scope='decoder3')
            logits = self.classify_block(decoder3)
            return logits
