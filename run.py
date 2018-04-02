import tensorflow as tf
from networks.relaynet_slim import ReLayNet
# from networks.networks import ReLayNet
# from networks.relaynet_slim import ReLayNet
# from data_loader.imdb import RetinalDataset
from data_loader.dataset import RetinalDataset
from models.relaynet_model import ReLayNetModel
from trainers.relaynet_trainer import ReLayNetTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
import numpy as np
from utils.utils_1 import get_args

def main():
    # capture the config path from the run arguments
    # then process the json configration file
    config = process_config('F:/deep_learning/relaynet_tensorflow/configs/relaynet.json')

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    config.learning_rate = np.concatenate((1e-1 * np.ones(20), 1e-2 * np.ones(20), 1e-2 * np.ones(20)))
    # network_fn = ReLayNet(10, config=config, scope='relaynet02')
    network_fn = ReLayNet(config=config)
    # create instance of the model you want
    model = ReLayNetModel(config, network_fn)
    # create your data generator
    data = RetinalDataset(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and path all previous components to it
    trainer = ReLayNetTrainer(sess, model, data, config, logger)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
