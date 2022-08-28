
from easydict import EasyDict
import torch
import numpy as np
import math

cfg = EasyDict()

# device (multi-core train is not supported)
cfg.device = torch.device("cuda:0")
cfg.visible = "0"

# lidar channel
cfg.lidar_channel = 32

# directory
cfg.oxford_home = '/home/jnu-ie/Dataset/oxford/'
cfg.proj_home = '/home/jnu-ie/repository/calib-using-voxel/'
cfg.model = 'oxford-20-1.5'

cfg.logdir = cfg.proj_home + 'checkpoint/' + cfg.model + '/'
cfg.data_subdir = 'gendata/data_oxford_5/'
cfg.traintxt = 'oxford-train.txt'
cfg.valtxt = 'oxford-val.txt'

# RoI
cfg.x_roi = [-25, 25]
cfg.y_roi = [-25, 25]
cfg.z_roi = [-1, 2]

# voxel
cfg.voxelsize = (0.05, 0.05, 0.05) # x y z
cfg.voxelrange = (-40.0, -40.0, -15.0, 40.0, 40.0, 15.0) # x y z
cfg.voxelshape = ( # z y x
    int((cfg.voxelrange[5] - cfg.voxelrange[2]) / cfg.voxelsize[2]),
    int((cfg.voxelrange[4] - cfg.voxelrange[1]) / cfg.voxelsize[1]),
    int((cfg.voxelrange[3] - cfg.voxelrange[0]) / cfg.voxelsize[0]))
cfg.voxelshape = np.array(cfg.voxelshape)
cfg.maxpoints = 3
cfg.maxvoxels = 60000
cfg.left_local = 224
cfg.right_local = 288

# train
cfg.max_left_rot_err = 20
cfg.max_left_trs_err = 1.5
cfg.num_cpu = 8
cfg.batchsize = 8
cfg.learing_rate = 5e-4
cfg.maxepoch = 60
cfg.MultiStepLR_milstone = [30, 40]
cfg.MultiStepLR_gamma = 0.5

# val
cfg.test_epochmodel = 60
