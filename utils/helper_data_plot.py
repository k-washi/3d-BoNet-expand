import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)

import numpy as np
import scipy.io
import copy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # include matpltlib
import open3d as o3d  # ver__0.8
import math
import random
import colorsys

from utils.logger import set_logger

logger = set_logger(__name__)




class Plot(object):
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]  # [(0.15, 1, 1.0), ..., (0.95, 1, 1.0)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        # Color情報なし
        if pc_xyzrgb.shape[1] == 3:
            o3d.visualization.draw_geometries([pc])
            return 0

        # Color情報あり
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = o3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        o3d.visualization.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_semins(pc_xyz, pc_semins, fix_color_num=None):

        #色の数を決定
        if fix_color_num is not None:
            ins_colors = Plot.random_colors(fix_color_num + 1, seed=2)
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_semins)) + 1, seed=2)  # cls 14

        ##############################
        #BBoxを作成する
        semins_labels = np.unique(pc_semins)
        semins_bbox = []
        Y_colors = np.zeros((pc_semins.shape[0], 3))
        for id, semins in enumerate(semins_labels):
            valid_ind = np.argwhere(pc_semins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if fix_color_num is not None:
                    semins = math.floor(semins % fix_color_num)
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            """
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            semins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])
            """

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins


if __name__ == "__main__":
    mplt = Plot()
    print(Plot.random_colors(5))  # [(0.0, 0.699, 1.0), ... , (0.5, 0.0, 1.0)]
