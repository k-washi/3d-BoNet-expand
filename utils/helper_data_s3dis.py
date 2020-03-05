import glob
import numpy as np
import random
import copy
from random import shuffle
import h5py
import os
import sys

from utils.logger import set_logger

logger = set_logger(__name__)

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)


class Data_Configs(object):
    sem_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
    sem_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    points_cc = 9
    sem_num = len(sem_names)
    ins_max_num = 24
    train_pts_num = 4096
    test_pts_num = 4096


class Data_S3DIS(object):
    """
    http://buildingparser.stanford.edu/dataset.html
    """

    def __init__(self, dataset_path, train_areas, test_areas, train_batch_size=4):
        self.root_folder_4_traintest = dataset_path
        self.train_files = self.load_full_file_list(train_areas)
        self.test_files = self.load_full_file_list(test_areas)
        logger.info("train files: {}".format(len(self.train_files)))
        logger.info("test files: {}".format(len(self.test_files)))

        self.ins_max_num = Data_Configs.ins_max_num
        self.train_batch_size = train_batch_size
        self.total_train_batch_num = len(self.train_files) // self.train_batch_size

        self.train_next_bat_index = 0

    def get_file_path(self, idx, train=True):
        if train:
            fs = self.train_files
        else:
            fs = self.test_files

        if len(fs) > idx:
            return fs[idx]
        else:
            return None

    def load_full_file_list(self, areas, check=False):
        all_files = []
        for a in areas:
            data_regix = os.path.join(self.root_folder_4_traintest, a + '*.h5')
            files = sorted(glob.glob((data_regix)))
            for f in files:
                fin = h5py.File(f, 'r')
                coords = fin['coords'][:]  # (35, 4096, 3)

                # 初めてデータを読み込む時はチェックが必要だが、以降は必要ない
                if check:
                    semIns_labels = fin['labels'][:].reshape([-1, 2])  # (35, 4096, 2) => (143360, 2)
                    ins_labels = semIns_labels[:, 1]  # instance label
                    sem_labels = semIns_labels[:, 0]  # semantic label

                    data_valid = True
                    ins_idx = np.unique(ins_labels)  # [ 0  1  2 ... 21 22]
                    for i_i in ins_idx:
                        if i_i <= -1: continue
                        sem_labels_tp = sem_labels[ins_labels == i_i]
                        unique_sem_labels = np.unique(sem_labels_tp)

                        # 一つの物体に2つ以上のセマンティックが割り当てられている場合
                        if len(unique_sem_labels) >= 2:
                            print('>= 2 sem for an ins:', f)
                            data_valid = False
                            break

                    if not data_valid:
                        continue

                block_num = coords.shape[0]
                for b in range(block_num):
                    all_files.append(f + '_' + str(b).zfill(4))  # [Area_1_WC_1.h5_0000, Area_1_WC_1.h5_0001, ...]

        return all_files

    @staticmethod
    def load_raw_data_file_s3dis_block(file_path):
        """
        :param file_path:
        :return:
            pc: 点群
            sem_labels: セマンティックラベル
            ins_labels: インスタンスラベル
        """
        block_id = int(file_path[-4:])
        file_path = file_path[0:-5]

        fin = h5py.File(file_path, 'r')
        coords = fin['coords'][block_id]
        points = fin['points'][block_id]
        semIns_labels = fin['labels'][block_id]

        pc = np.concatenate([coords, points[:, 3:9]], axis=-1)
        sem_labels = semIns_labels[:, 0]
        ins_labels = semIns_labels[:, 1]

        return pc, sem_labels, ins_labels

    @staticmethod
    def data_plot(file_path, mode=0):
        """
        3dでデータをプロット
        :param file_path:
        :param mode: 0: non color, 1: sem label, 2: ins label
        :return:
        """
        block_id = int(file_path[-4:])
        file_path = file_path[0:-5]

        fin = h5py.File(file_path, 'r')
        coords = fin['coords'][block_id]
        points = fin['points'][block_id]

        pc = np.concatenate([coords, points[:, 3:9]], axis=-1)

        from utils.helper_data_plot import Plot as Plot
        if mode == 0:
            Plot.draw_pc(pc)
        else:
            semIns_labels = fin['labels'][block_id]
            if mode == 1:
                sem_labels = semIns_labels[:, 0]
                Plot.draw_pc_semins(pc, pc_semins=sem_labels, fix_color_num=13)
            elif mode == 2:
                ins_labels = semIns_labels[:, 1]
                Plot.draw_pc_semins(pc, pc_semins=ins_labels)
            else:
                logger.error("mode is incorrect, mode = {1, 2, 3}")

    @staticmethod
    def get_bbvert_pmask_labels(pc, ins_labels):
        """

        :param pc:
        :param ins_labels:
        :return:
        """
        gt_bbvert_padded = np.zeros((Data_Configs.ins_max_num, 2, 3), dtype=np.float32)  # (インスタンス最大数, 最小最大, xyz )
        gt_pmask = np.zeros((Data_Configs.ins_max_num, pc.shape[0]), dtype=np.float32)  # (インスタンス最大数, 点群数)
        count = -1
        unique_ins_labels = np.unique(ins_labels)
        for ins_ind in unique_ins_labels:
            if ins_ind <= -1:
                continue

            # インスタンスの数が超えたら無視
            count += 1
            if count >= Data_Configs.ins_max_num:
                print('ignored! more than max instances:', len(unique_ins_labels))
                continue

            # ins_indに一致するラベルを取得する
            ins_labels_tp = np.zeros(ins_labels.shape, dtype=np.int8)
            ins_labels_tp[ins_labels == ins_ind] = 1
            ins_labels_tp = np.reshape(ins_labels_tp, [-1])
            gt_pmask[count, :] = ins_labels_tp # gt_pmask: ins_indに一致する点を1, それ以外を0にする

            ins_labels_tp_ind = np.argwhere(ins_labels_tp == 1)
            ins_labels_tp_ind = np.reshape(ins_labels_tp_ind, [-1])

            # bb min_xyz, max_xyz
            pc_xyz_tp = pc[:, 0:3]
            pc_xyz_tp = pc_xyz_tp[ins_labels_tp_ind]
            gt_bbvert_padded[count, 0, 0] = np.min(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 0, 1] = np.min(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 0, 2] = np.min(pc_xyz_tp[:, 2])
            gt_bbvert_padded[count, 1, 0] = np.max(pc_xyz_tp[:, 0])
            gt_bbvert_padded[count, 1, 1] = np.max(pc_xyz_tp[:, 1])
            gt_bbvert_padded[count, 1, 2] = np.max(pc_xyz_tp[:, 2])

        return gt_bbvert_padded, gt_pmask

    @staticmethod
    def load_fixed_points(file_path):
        pc_xyzrgb, sem_labels, ins_labels = Data_S3DIS.load_raw_data_file_s3dis_block(file_path)

        # center xy within the block
        min_x, max_x = np.min(pc_xyzrgb[:, 0]), np.max(pc_xyzrgb[:, 0])
        min_y, max_y = np.min(pc_xyzrgb[:, 1]), np.max(pc_xyzrgb[:, 1])
        min_z, max_z = np.min(pc_xyzrgb[:, 2]), np.max(pc_xyzrgb[:, 2])

        ori_xyz = copy.deepcopy(pc_xyzrgb[:, 0:3])  # reserved for final visualization
        use_zero_one_center = True
        if use_zero_one_center:
            pc_xyzrgb[:, 0] = (pc_xyzrgb[:, 0] - min_x)/ np.maximum((max_x - min_x), 1e-3)
            pc_xyzrgb[:, 1] = (pc_xyzrgb[:, 1] - min_y)/ np.maximum((max_y - min_y), 1e-3)
            pc_xyzrgb[:, 2] = (pc_xyzrgb[:, 2] - min_z)/ np.maximum((max_z - min_z), 1e-3)
        pc_xyzrgb = np.concatenate([pc_xyzrgb, ori_xyz], axis=-1)  # (4096, 12) [:9]=>ref point, [9:]:origin point

        ########
        sem_labels = sem_labels.reshape([-1]) # (全てのラベル数 4096, )に圧縮
        ins_labels = ins_labels.reshape([-1]) # (全てのインスタンス数 4096, )に圧縮
        bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.get_bbvert_pmask_labels(pc_xyzrgb, ins_labels)

        # 各点に対してone hotなラベル(sem)を与える
        psem_onehot_labels = np.zeros((pc_xyzrgb.shape[0], Data_Configs.sem_num), dtype=np.int8)  # (4096, 13)
        for idx, s in enumerate(sem_labels):
            if sem_labels[idx] == -1:
                continue  # invalid points
            sem_idx = Data_Configs.sem_ids.index(s)
            psem_onehot_labels[idx, sem_idx] = 1

        return pc_xyzrgb, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels

    def load_train_next_batch(self):
        """
        次の訓点データセットを読み込む
        :return:
        """
        bat_files = self.train_files[self.train_next_bat_index * self.train_batch_size:(self.train_next_bat_index + 1) * self.train_batch_size]
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for file in bat_files:
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels \
                = Data_S3DIS.load_fixed_points(file)

            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        self.train_next_bat_index += 1
        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels

    def load_test_next_batch_random(self):
        idx = random.sample(range(len(self.test_files)), self.train_batch_size)
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for i in idx:
            file = self.test_files[i]
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.load_fixed_points(
                file)
            bat_pc.append(pc)
            bat_sem_labels.append(sem_labels)
            bat_ins_labels.append(ins_labels)
            bat_psem_onehot_labels.append(psem_onehot_labels)
            bat_bbvert_padded_labels.append(bbvert_padded_labels)
            bat_pmask_padded_labels.append(pmask_padded_labels)

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels

    def load_test_next_batch_sq(self, bat_files):
        """
        訓練用のデータを一つ取り出す。
        :param bat_files:
        :return:
        """
        bat_pc = []
        bat_sem_labels = []
        bat_ins_labels = []
        bat_psem_onehot_labels = []
        bat_bbvert_padded_labels = []
        bat_pmask_padded_labels = []
        for file in bat_files:
            pc, sem_labels, ins_labels, psem_onehot_labels, bbvert_padded_labels, pmask_padded_labels = Data_S3DIS.load_fixed_points(
                file)
            bat_pc += [pc]
            bat_sem_labels += [sem_labels]
            bat_ins_labels += [ins_labels]
            bat_psem_onehot_labels += [psem_onehot_labels]
            bat_bbvert_padded_labels += [bbvert_padded_labels]
            bat_pmask_padded_labels += [pmask_padded_labels]

        bat_pc = np.asarray(bat_pc, dtype=np.float32)
        bat_sem_labels = np.asarray(bat_sem_labels, dtype=np.float32)
        bat_ins_labels = np.asarray(bat_ins_labels, dtype=np.float32)
        bat_psem_onehot_labels = np.asarray(bat_psem_onehot_labels, dtype=np.float32)
        bat_bbvert_padded_labels = np.asarray(bat_bbvert_padded_labels, dtype=np.float32)
        bat_pmask_padded_labels = np.asarray(bat_pmask_padded_labels, dtype=np.float32)

        return bat_pc, bat_sem_labels, bat_ins_labels, bat_psem_onehot_labels, bat_bbvert_padded_labels, bat_pmask_padded_labels, bat_files

    def shuffle_train_files(self, ep):
        index = list(range(len(self.train_files)))
        random.seed(ep)
        shuffle(index)
        train_files_new = []
        for i in index:
            train_files_new.append(self.train_files[i])
        self.train_files = train_files_new
        self.train_next_bat_index = 0
        print('train files shuffled!')


if __name__ == "__main__":
    # train_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6']
    train_areas = ['Area_1']
    test_areas = ['Area_5']

    dataset_path = '/Users/washizakikai/data/Data_S3DIS'

    Data = Data_S3DIS(dataset_path, train_areas, test_areas)
    file_path = Data.get_file_path(1)
    pc, sl, il, sol, bbl, pml = Data.load_fixed_points(file_path)
    print(pc.shape)
    print(sl.shape)
    print(il.shape)
    print(sol.shape)
    print(bbl.shape)
    print(pml.shape)
    # print(file_path)
    # Data.data_plot(file_path, mode=1)
