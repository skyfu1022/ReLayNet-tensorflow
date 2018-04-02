from base.base_model import BaseModel
import tensorflow as tf
from utils.ops import dice_loss, mce_loss, f1_score_metrix


class ReLayNetModel(BaseModel):
    def __init__(self, config, network_fn):
        super(ReLayNetModel, self).__init__(config)
        self.network_fn = network_fn
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.learning_rate = tf.placeholder(tf.float32)

        self.images = tf.placeholder(tf.float32, shape=[self.config.batch_size] + self.config.image_size)
        self.labels = tf.placeholder(tf.int64, shape=[self.config.batch_size] + self.config.image_size)
        self.weights = tf.placeholder(tf.float32, shape=[self.config.batch_size] + self.config.image_size)
        self.one_hot_label = tf.squeeze(tf.one_hot(self.labels, self.config.num_classes, dtype=tf.float32), [-2])
        # network_architecture
        logits = self.network_fn(self.images)
        self.pred = tf.argmax(logits, -1)
        self.float_weights = tf.to_float(self.weights)
        # self.flat_labels = tf.reshape(one_hot_label, [-1, self.config.num_classes])
        # self.flat_logits = tf.reshape(logits, [-1, self.config.num_classes])
        self.probs = tf.nn.softmax(logits)
        # self.flat_weithts = tf.to_float(tf.reshape(self.weights, [-1]))
        # self.flat_weights = tf.reshape(self.weights, [-1, 1])
        one_hot_pred = tf.one_hot(self.pred, self.config.num_classes, dtype=tf.float32)
        self.losses = self.total_loss()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.config.momentum)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.losses, global_step=self.global_step_tensor)
        correct_prediction = tf.equal(self.pred, tf.squeeze(self.labels))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.f1_score = f1_score_metrix(self.one_hot_label, one_hot_pred, self.config.num_classes)

    def total_loss(self):
        with tf.variable_scope('loss'):
            self.dice = dice_loss(self.probs, self.one_hot_label)
            self.mce = mce_loss(self.probs, self.one_hot_label, self.weights)
            self.l2 = tf.losses.get_regularization_loss()
            total_losses = self.config.alpha * self.mce + self.config.beta * self.dice + self.l2
            return total_losses



    def init_saver(self):
        # here you initalize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
