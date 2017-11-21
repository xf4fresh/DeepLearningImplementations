#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
sudo pip install colorama==0.3.9
sudo pip install terminaltables==3.1.0
"""

import os
import sys
# Disable Tensorflow's INFO and WARNING messages
# See http://stackoverflow.com/questions/35911252
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import flags
import tensorflow as tf
import train
import train_mnist

FLAGS = tf.app.flags.FLAGS


def launch_training():
    # train.train_model()
    train_mnist.train_model()


def main(argv=None):
    assert FLAGS.run in ["train", "inference"], "Choose [train|inference]"

    if FLAGS.run == 'train':
        launch_training()


if __name__ == '__main__':
    sys.argv = ['', '--run', 'train', '--nb_epoch', '20', '--random_seed', '2017', '--data_format', 'NHWC']
    flags.define_flags()  # 设置默认值
    tf.app.run()
