
import torch.utils.data as data
from src.utils import *
import random

import numpy as np
import torch


class DataFactory(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        if mode == 'train':
            with open(cfg.proj_home + 'gendata/' + cfg.traintxt) as f:
                self.datalist = f.readlines()
        elif mode == 'val':
            with open(cfg.proj_home + 'gendata/' + cfg.valtxt) as f:
                self.datalist = f.readlines()

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        while True:
            seq, imgnum = self.datalist[idx].split(' ')
            imgnum = imgnum.strip()
            sgm_path = self.cfg.proj_home + self.cfg.data_subdir + seq + '/disp_' + str(self.cfg.sgm_path) + '/' + imgnum + '.bin'
            velo_path = self.cfg.proj_home + self.cfg.data_subdir + seq + '/velodyne_' + str(self.cfg.lidar_channel) + '/' + imgnum + '.bin'
            calib_path = self.cfg.odometry_home + self.cfg.calib_subdir + seq + '/calib.txt'

            # file load
            sgmroad = np.fromfile(sgm_path, dtype=np.float32).reshape((-1, 3))
            veloroad = np.fromfile(velo_path, dtype=np.float32).reshape((-1, 3))

            if (veloroad.shape[0] == 0 or sgmroad.shape[0] == 0) and self.mode == 'train':
                idx = random.randint(0, len(self.datalist) - 1)
                continue
            elif (veloroad.shape[0] == 0 or sgmroad.shape[0] == 0) and self.mode == 'val':
                assert 'datafactory error'
            break

        # calibration
        # refer : https://github.com/utiasSTARS/pykitti
        filedata = read_calib_file(calib_path)
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))

        T_cam0_velo = np.reshape(filedata['Tr'], (3, 4))
        T_cam0_velo = np.vstack([T_cam0_velo, [0, 0, 0, 1]])
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        matrix_R = T2.dot(T_cam0_velo)
        veloroad = rot_and_trs_points(veloroad, matrix_R)

        # gen error
        # refer : https://github.com/LvXudong-HIT/LCCNet
        sgmRmat = np.zeros((4, 4), dtype=np.float32)
        sgmRmat[0, 0] = 1
        sgmRmat[1, 1] = 1
        sgmRmat[2, 2] = 1
        sgmRmat[3, 3] = 1

        maxvelorot = self.cfg.max_velo_rot_err
        maxvelotrs = self.cfg.max_velo_trs_err
        velo_rotx = np.random.uniform(-maxvelorot, maxvelorot) * (3.141592 / 180.0)
        velo_roty = np.random.uniform(-maxvelorot, maxvelorot) * (3.141592 / 180.0)
        velo_rotz = np.random.uniform(-maxvelorot, maxvelorot) * (3.141592 / 180.0)
        velo_trsx = np.random.uniform(-maxvelotrs, maxvelotrs)
        velo_trsy = np.random.uniform(-maxvelotrs, maxvelotrs)
        velo_trsz = np.random.uniform(-maxvelotrs, maxvelotrs)

        velorotmat = eulerAnglesToRotationMatrix([velo_rotx, velo_roty, velo_rotz])
        veloRmat = np.zeros((4, 4), dtype=np.float32)
        veloRmat[:3, :3] = velorotmat[:3,:3]
        veloRmat[0, 3] = velo_trsx
        veloRmat[1, 3] = velo_trsy
        veloRmat[2, 3] = velo_trsz
        veloRmat[3, 3] = 1

        veloroad = rot_and_trs_points(veloroad, veloRmat)

        # gen target error
        tgtRmat = np.linalg.inv(veloRmat) @ sgmRmat
        targetQuaternion = quaternion_from_rotation_matrix(tgtRmat[:3, :3])
        targetTranslate = np.array((tgtRmat[0, 3], tgtRmat[1, 3], tgtRmat[2, 3]))

        if self.mode == 'val':
            return sgmroad, veloroad, tgtRmat
        return sgmroad, veloroad, targetQuaternion, targetTranslate
    
    def collate_fn_cpu(self, batch):
        sgm_list = []
        velo_list = []
        tgtQ_list = []
        tgtT_list = []

        for sgmroad, veloroad, targetQuaternion, targetTranslate in batch:
            sgm_list.append(torch.from_numpy(sgmroad))
            velo_list.append(torch.from_numpy(veloroad))
            tgtQ_list.append(torch.from_numpy(targetQuaternion).unsqueeze(0))
            tgtT_list.append(torch.from_numpy(targetTranslate).unsqueeze(0))
            continue
        tgtQ_list = torch.cat(tgtQ_list, dim=0)
        tgtT_list = torch.cat(tgtT_list, dim=0)
        traindatadict = {
            'sgm': sgm_list,
            'velo': velo_list,
            'tgtQ': tgtQ_list,
            'tgtT': tgtT_list
        }

        return traindatadict
