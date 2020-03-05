import os
import sys

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from utils.logger import set_logger

logger = set_logger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)


class Ops(object):

    @staticmethod
    def lrelu(x, leak=0.2):
        return tf.nn.leaky_relu(x, alpha=0.2)