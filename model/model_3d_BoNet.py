import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)

import shutil

import tensorflow as tf


from utils.logger import set_logger

logger = set_logger(__name__)




class BoNet(object):
    def __init__(self, configs):
        self.points_cc = configs.points_cc
        self.sem_num = configs.sem_num  #セマンティックラベル数
        self.bb_num = configs.ins_max_num  #インスタンスの最大数


    def backbone_pointnet2(selfself, X_pc, is_train=None):
        """
        PointNet++の実装
        :param X_pc:
        :param is_train:
        :return:
        """




