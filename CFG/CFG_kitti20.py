
from easydict import EasyDict
import torch
import numpy as np
import math

cfg = EasyDict()

# device (multi-core train is not supported)
cfg.device = torch.device("cuda:0")
cfg.visible = "0"

# sgm path direction
cfg.sgm_path = 4
# lidar channel
cfg.lidar_channel = 64

# directory
cfg.odometry_home = '/home/jnu-ie/Dataset/odometry/'
cfg.proj_home = '/home/jnu-ie/repository/calib-using-voxel/'
cfg.model = 'kitti-20-1.5'

cfg.logdir = cfg.proj_home + 'checkpoint/' + cfg.model + '/'
cfg.data_subdir = 'gendata/data_odometry_20/dataset/sequences/'
cfg.calib_subdir = 'data_odometry_color/dataset/sequences/'
cfg.traintxt = 'kitti-train.txt'
cfg.valtxt = 'kitti-val.txt'

# RoI
cfg.camx_roi = [-10, 10]
cfg.camy_roi = [-1, 2]
cfg.camz_roi = [0, 50]

cfg.lidarx_roi = [0, 50]
cfg.lidary_roi = [-10, 10]
cfg.lidarz_roi = [-2, 1]

# voxel
cfg.voxelsize = (0.05, 0.05, 0.05) # x y z
cfg.voxelrange = (-15.0, -15.0, 0, 15.0, 15.0, 55.0) # x y z
cfg.voxelshape = ( # z y x
    int((cfg.voxelrange[5] - cfg.voxelrange[2]) / cfg.voxelsize[2]),
    int((cfg.voxelrange[4] - cfg.voxelrange[1]) / cfg.voxelsize[1]),
    int((cfg.voxelrange[3] - cfg.voxelrange[0]) / cfg.voxelsize[0]))
cfg.voxelshape = np.array(cfg.voxelshape)
cfg.maxpoints = 3
cfg.maxvoxels = 60000
cfg.sgm_local = 96
cfg.velo_local = 160
#cfg.sgm_local = 32
#cfg.velo_local = 128

# train
cfg.max_velo_rot_err = 20
cfg.max_velo_trs_err = 1.5
cfg.num_cpu = 8
cfg.batchsize = 8
cfg.learing_rate = 5e-4
cfg.maxepoch = 60
cfg.MultiStepLR_milstone = [30, 40]
cfg.MultiStepLR_gamma = 0.5

# val
cfg.test_epochmodel = 60
