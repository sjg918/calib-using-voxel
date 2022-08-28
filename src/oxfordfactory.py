
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
            leftnum, rightnum = self.datalist[idx].split(' ')
            rightnum = rightnum.strip()

            if self.mode == 'train':
                left_path = self.cfg.proj_home + self.cfg.data_subdir + 'scene1/left/' + leftnum + '.bin' 
                right_path = self.cfg.proj_home + self.cfg.data_subdir + 'scene1/right/' + rightnum + '.bin' 
            elif self.mode == 'val':
                left_path = self.cfg.proj_home + self.cfg.data_subdir + 'scene2/left/' + leftnum + '.bin' 
                right_path = self.cfg.proj_home + self.cfg.data_subdir + 'scene2/right/' + rightnum + '.bin' 

            # file load
            leftbin = np.fromfile(left_path, dtype=np.float32).reshape((-1, 3))
            rightbin = np.fromfile(right_path, dtype=np.float32).reshape((-1, 3))

            if (leftbin.shape[0] == 0 or rightbin.shape[0] == 0) and self.mode == 'train':
                idx = random.randint(0, len(self.datalist) - 1)
                continue
            elif (leftbin.shape[0] == 0 or rightbin.shape[0] == 0) and self.mode == 'val':
                assert 'datafactory error'
            break

        # gen error
        # refer : https://github.com/LvXudong-HIT/LCCNet
        maxleftrot = self.cfg.max_left_rot_err
        maxlefttrs = self.cfg.max_left_trs_err
        left_rotx = np.random.uniform(-maxleftrot, maxleftrot) * (3.141592 / 180.0)
        left_roty = np.random.uniform(-maxleftrot, maxleftrot) * (3.141592 / 180.0)
        left_rotz = np.random.uniform(-maxleftrot, maxleftrot) * (3.141592 / 180.0)
        left_trsx = np.random.uniform(-maxlefttrs, maxlefttrs)
        left_trsy = np.random.uniform(-maxlefttrs, maxlefttrs)
        left_trsz = np.random.uniform(-maxlefttrs, maxlefttrs)

        leftrotmat = eulerAnglesToRotationMatrix([left_rotx, left_roty, left_rotz])
        leftRmat = np.zeros((4, 4), dtype=np.float32)
        leftRmat[:3, :3] = leftrotmat[:3,:3]
        leftRmat[0, 3] = left_trsx
        leftRmat[1, 3] = left_trsy
        leftRmat[2, 3] = left_trsz
        leftRmat[3, 3] = 1

        leftbin = rot_and_trs_points(leftbin, leftRmat)

        # gen target error
        tgtRmat = np.linalg.inv(leftRmat)
        targetQuaternion = quaternion_from_rotation_matrix(tgtRmat[:3, :3])
        targetTranslate = np.array((tgtRmat[0, 3], tgtRmat[1, 3], tgtRmat[2, 3]))

        if self.mode == 'val':
            return rightbin, leftbin, tgtRmat
        return rightbin, leftbin, targetQuaternion, targetTranslate
    
    def collate_fn_cpu(self, batch):
        right_list = []
        left_list = []
        tgtQ_list = []
        tgtT_list = []

        for rightbin, leftbin, targetQuaternion, targetTranslate in batch:
            right_list.append(torch.from_numpy(rightbin))
            left_list.append(torch.from_numpy(leftbin))
            tgtQ_list.append(torch.from_numpy(targetQuaternion).unsqueeze(0))
            tgtT_list.append(torch.from_numpy(targetTranslate).unsqueeze(0))
            continue
        tgtQ_list = torch.cat(tgtQ_list, dim=0)
        tgtT_list = torch.cat(tgtT_list, dim=0)
        traindatadict = {
            'right': right_list,
            'left': left_list,
            'tgtQ': tgtQ_list,
            'tgtT': tgtT_list
        }

        return traindatadict
