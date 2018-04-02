from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
from utils.ops import cal_weight


class ReLayNetTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ReLayNetTrainer, self).__init__(sess, model, data, config, logger)
        self.train_batch = self.data.get_batch('train')
        self.val_batch = self.data.get_batch('val')

    def eval_fun(self, func, num_samples, cur_epoch, mode):
        loop = tqdm(range(num_samples // self.config.batch_size))
        losses = []
        accs = []
        mces = []
        dices = []
        l2s = []
        f1_scores = []
        for it in loop:
            loss, mce, dice, l2, acc, f1_score = func()
            losses.append(loss)
            accs.append(acc)
            mces.append(mce)
            dices.append(dice)
            l2s.append(l2)
            f1_scores.append(f1_score)
        loss = np.mean(losses)
        acc = np.mean(accs)
        mce = np.mean(mces)
        dice = np.mean(dices)
        l2 = np.mean(l2s)
        f1_score = np.mean(f1_scores)
        per_class_fscore = np.mean(f1_scores, axis=0)
        # cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        summaries_dict['acc'] = acc
        summaries_dict['mce'] = mce
        summaries_dict['dice'] = dice
        summaries_dict['l2'] = l2
        summaries_dict['f1_score'] = f1_score
        for i in range(len(per_class_fscore)):
            summaries_dict['f1_score_%d' % i] = per_class_fscore[i]
        print("%s epoch %d|| loss: %f, acc: %f, f1 score: %f" % (mode, cur_epoch, loss, acc, f1_score))
        self.logger.summarize(cur_epoch, summerizer=mode, summaries_dict=summaries_dict)

    def train_epoch(self):
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        self.cur_learning_rate = self.config.learning_rate[cur_epoch]
        self.eval_fun(self.train_step, self.data.num_train_samples, cur_epoch, 'train')
        self.eval_fun(self.eval, self.data.num_val_samples, cur_epoch, 'val')
        self.model.save(self.sess)

    def train_step(self):
        image_batch, label_batch, contour_batch = self.sess.run(self.train_batch)
        image_batch = image_batch / 127.5 - 1.0
        weight_batch = cal_weight(label_batch, contour_batch, self.config.w1, self.config.w2)
        fetches = [self.model.train_step, self.model.losses, self.model.mce, self.model.dice, self.model.l2, self.model.accuracy, self.model.f1_score]
        feed_dict = {self.model.images: image_batch, self.model.labels: label_batch, self.model.weights: weight_batch,
                     self.model.is_training: True, self.model.learning_rate: self.cur_learning_rate}
        _, loss, mce, dice, l2, acc, f1_score = self.sess.run(fetches, feed_dict=feed_dict)
        return loss, mce, dice, l2, acc, f1_score

    def eval(self):
        image_batch, label_batch, weight_batch = self.sess.run(self.val_batch)
        image_batch = image_batch / 127.5 - 1.0
        # weight_batch = cal_weight(label_batch, contour_batch, self.config.w1, self.config.w2)
        fetches = [self.model.losses, self.model.mce, self.model.dice, self.model.l2, self.model.accuracy, self.model.f1_score]
        feed_dict = {self.model.images: image_batch, self.model.labels: label_batch, self.model.weights: weight_batch,
                     self.model.is_training: False}
        loss, mce, dice, l2, acc, f1_score = self.sess.run(fetches, feed_dict=feed_dict)
        return loss, mce, dice, l2, acc, f1_score
