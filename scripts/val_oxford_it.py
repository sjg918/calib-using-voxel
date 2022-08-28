
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import datetime
import random
import cv2
import time

import torch
import torch.nn.functional as F
from spconv.pytorch.utils import PointToVoxel
import spconv.pytorch as spconv
import sys
import shutil
import matplotlib.pyplot as plt
from pykitti import odometry

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from CFG.CFG_oxford20 import cfg
from src.voxelnet import Deepv2_base
from src.losses import quaternion_distance
from src.oxfordfactory import DataFactory
from src.utils import *

def val_oxford_iterative():
    print(cfg.model)
    
    model20 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    model10 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    model05 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    model02 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    model01 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    
    load_model(model20,cfg.proj_home + 'checkpoint/oxford-20.pth', cfg.device)
    load_model(model10,cfg.proj_home + 'checkpoint/oxford-10.pth', cfg.device)
    load_model(model05,cfg.proj_home + 'checkpoint/oxford-5.pth', cfg.device)
    load_model(model02,cfg.proj_home + 'checkpoint/oxford-2.pth', cfg.device)
    load_model(model01,cfg.proj_home + 'checkpoint/oxford-1-0.1-(0.5,5).pth', cfg.device)

    point2voxel20 = PointToVoxel(vsize_xyz=cfg.voxelsize,
                           coors_range_xyz=cfg.voxelrange,
                           num_point_features=4,
                           max_num_voxels=cfg.maxvoxels,
                           max_num_points_per_voxel=cfg.maxpoints,
                           device=cfg.device)

    # define dataloader
    oxford_dataset = DataFactory(cfg, 'val')

    model20.eval()
    model10.eval()
    model05.eval()
    model02.eval()
    model01.eval()

    error_Roll_20 = []
    error_Pitch_20 = []
    error_Yaw_20 = []
    error_X_20 = []
    error_Y_20 = []
    error_Z_20 = []
    
    error_Roll_10 = []
    error_Pitch_10 = []
    error_Yaw_10 = []
    error_X_10 = []
    error_Y_10 = []
    error_Z_10 = []
    
    error_Roll_05 = []
    error_Pitch_05 = []
    error_Yaw_05 = []
    error_X_05 = []
    error_Y_05 = []
    error_Z_05 = []
    
    error_Roll_02 = []
    error_Pitch_02 = []
    error_Yaw_02 = []
    error_X_02 = []
    error_Y_02 = []
    error_Z_02 = []
    
    error_Roll_01 = []
    error_Pitch_01 = []
    error_Yaw_01 = []
    error_X_01 = []
    error_Y_01 = []
    error_Z_01 = []

    with torch.no_grad():
        for cnt in range(len(oxford_dataset)):
            rightbin, leftbin, misRTmat = oxford_dataset[cnt]
            misRTmat = np.linalg.inv(misRTmat)
                        
            # forward(20)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(rightbin, leftbin, point2voxel20, cfg)
            quaternion, translate = model20(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu().numpy()
            
            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, misRTmat)            
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_20.append(error_rot[0])
            error_Pitch_20.append(error_rot[1])
            error_Yaw_20.append(error_rot[2])
            error_X_20.append(error_trs[0])
            error_Y_20.append(error_trs[1])
            error_Z_20.append(error_trs[2])

            # calib 20
            calib_RT = np.zeros((4,4), dtype=np.float32)
            calib_RT[:3, :3] = rot_matrix
            calib_RT[0, 3] = translate[0]
            calib_RT[1, 3] = translate[1]
            calib_RT[2, 3] = translate[2]
            calib_RT[3, 3] = 1
            run_calib_RT = calib_RT
            newleftbin = rot_and_trs_points(leftbin, run_calib_RT)
            
            # forward(10)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(rightbin, newleftbin, point2voxel20, cfg)
            quaternion, translate = model10(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu()

            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, run_calib_RT @ misRTmat)            
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_10.append(error_rot[0])
            error_Pitch_10.append(error_rot[1])
            error_Yaw_10.append(error_rot[2])
            error_X_10.append(error_trs[0])
            error_Y_10.append(error_trs[1])
            error_Z_10.append(error_trs[2])

            # calib 10
            calib_RT = np.zeros((4,4), dtype=np.float32)
            calib_RT[:3, :3] = rot_matrix
            calib_RT[0, 3] = translate[0]
            calib_RT[1, 3] = translate[1]
            calib_RT[2, 3] = translate[2]
            calib_RT[3, 3] = 1
            run_calib_RT = calib_RT @ run_calib_RT
            newleftbin = rot_and_trs_points(leftbin, run_calib_RT)
                        
            # forward(05)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(rightbin, newleftbin, point2voxel20, cfg)
            quaternion, translate = model05(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu()
            
            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, run_calib_RT @ misRTmat)            
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_05.append(error_rot[0])
            error_Pitch_05.append(error_rot[1])
            error_Yaw_05.append(error_rot[2])
            error_X_05.append(error_trs[0])
            error_Y_05.append(error_trs[1])
            error_Z_05.append(error_trs[2])

            # calib 5
            calib_RT = np.zeros((4,4), dtype=np.float32)
            calib_RT[:3, :3] = rot_matrix
            calib_RT[0, 3] = translate[0]
            calib_RT[1, 3] = translate[1]
            calib_RT[2, 3] = translate[2]
            calib_RT[3, 3] = 1
            run_calib_RT = calib_RT @ run_calib_RT
            newleftbin = rot_and_trs_points(leftbin, run_calib_RT)
            
            # forward(02)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(rightbin, newleftbin, point2voxel20, cfg)
            quaternion, translate = model02(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu()
            
            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, run_calib_RT @ misRTmat)
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_02.append(error_rot[0])
            error_Pitch_02.append(error_rot[1])
            error_Yaw_02.append(error_rot[2])
            error_X_02.append(error_trs[0])
            error_Y_02.append(error_trs[1])
            error_Z_02.append(error_trs[2])

            # calib 2
            calib_RT = np.zeros((4,4), dtype=np.float32)
            calib_RT[:3, :3] = rot_matrix
            calib_RT[0, 3] = translate[0]
            calib_RT[1, 3] = translate[1]
            calib_RT[2, 3] = translate[2]
            calib_RT[3, 3] = 1
            run_calib_RT = calib_RT @ run_calib_RT
            newleftbin = rot_and_trs_points(leftbin, run_calib_RT)
            
            # forward(01)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(rightbin, newleftbin, point2voxel20, cfg)
            quaternion, translate = model01(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu()

            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, run_calib_RT @ misRTmat)
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_01.append(error_rot[0])
            error_Pitch_01.append(error_rot[1])
            error_Yaw_01.append(error_rot[2])
            error_X_01.append(error_trs[0])
            error_Y_01.append(error_trs[1])
            error_Z_01.append(error_trs[2])
            
            continue
            
    
    error_Roll = np.array(error_Roll_20)
    error_Pitch = np.array(error_Pitch_20)
    error_Yaw = np.array(error_Yaw_20)
    error_X = np.array(error_X_20)
    error_Y = np.array(error_Y_20)
    error_Z = np.array(error_Z_20)
    print_func("20", error_Roll, error_Pitch, error_Yaw, error_X, error_Y, error_Z)
    
    error_Roll = np.array(error_Roll_10)
    error_Pitch = np.array(error_Pitch_10)
    error_Yaw = np.array(error_Yaw_10)
    error_X = np.array(error_X_10)
    error_Y = np.array(error_Y_10)
    error_Z = np.array(error_Z_10)
    print_func("10", error_Roll, error_Pitch, error_Yaw, error_X, error_Y, error_Z)
    
    error_Roll = np.array(error_Roll_05)
    error_Pitch = np.array(error_Pitch_05)
    error_Yaw = np.array(error_Yaw_05)
    error_X = np.array(error_X_05)
    error_Y = np.array(error_Y_05)
    error_Z = np.array(error_Z_05)
    print_func("05", error_Roll, error_Pitch, error_Yaw, error_X, error_Y, error_Z)
    
    error_Roll = np.array(error_Roll_02)
    error_Pitch = np.array(error_Pitch_02)
    error_Yaw = np.array(error_Yaw_02)
    error_X = np.array(error_X_02)
    error_Y = np.array(error_Y_02)
    error_Z = np.array(error_Z_02)
    print_func("02", error_Roll, error_Pitch, error_Yaw, error_X, error_Y, error_Z)
    
    error_Roll = np.array(error_Roll_01)
    error_Pitch = np.array(error_Pitch_01)
    error_Yaw = np.array(error_Yaw_01)
    error_X = np.array(error_X_01)
    error_Y = np.array(error_Y_01)
    error_Z = np.array(error_Z_01)
    print_func("01", error_Roll, error_Pitch, error_Yaw, error_X, error_Y, error_Z)
        
    
def mk_sparsely_embedded_tensor(sgmroad, veloroad, point2voxel, config):
    sgmpoints = torch.from_numpy(sgmroad)
    sgmpoints = sgmpoints.to(config.device)
    velopoints = torch.from_numpy(veloroad)
    velopoints = velopoints.to(config.device)

    sgm_voxels, sgm_indices, sgm_num_p_in_vx = point2voxel(sgmpoints)

    sgm_voxels = sgm_voxels[:, :, :3].sum(dim=1, keepdim=False) / sgm_num_p_in_vx.type_as(sgm_voxels).view(-1, 1)
    sgm_batch = torch.zeros((sgm_num_p_in_vx.shape[0], 1), dtype=torch.int32, device=config.device)
    sgm_coors = sgm_indices.to(dtype=torch.int32)
    sgm_coors = torch.cat((sgm_batch, sgm_coors), dim=1)

    sgm_input = spconv.SparseConvTensor(sgm_voxels, sgm_coors, config.voxelshape, 1)

    velo_voxels, velo_indices, velo_num_p_in_vx = point2voxel(velopoints)

    velo_voxels = velo_voxels[:, :, :3].sum(dim=1, keepdim=False) / velo_num_p_in_vx.type_as(velo_voxels).view(-1, 1)
    velo_batch = torch.zeros((velo_num_p_in_vx.shape[0], 1), dtype=torch.int32, device=config.device)
    velo_coors = velo_indices.to(dtype=torch.int32)
    velo_coors = torch.cat((velo_batch, velo_coors), dim=1)

    velo_input = spconv.SparseConvTensor(velo_voxels, velo_coors, config.voxelshape, 1)
    return sgm_input, velo_input


def cal_error(rot_matrix, translate, misRTmat):
    predRTmat = np.zeros((4, 4), dtype=np.float32)
    predRTmat[:3, :3] = rot_matrix[:3,:3]
    predRTmat[0, 3] = translate[0]
    predRTmat[1, 3] = translate[1]
    predRTmat[2, 3] = translate[2]
    predRTmat[3, 3] = 1

    error_MAT = predRTmat @ misRTmat
    error_ROT = rotationMatrixToEulerAngles(error_MAT[:3, :3]) * (180.0 / np.pi)
    error_TRS = error_MAT[:3, 3]
    return error_ROT, error_TRS


def print_func(name, error_Roll, error_Pitch, error_Yaw, error_X, error_Y, error_Z):
    print("------------" + name + "------------")
    print("Roll : ", error_Roll.mean(), " ", error_Roll.std(), " ", error_Roll.max())
    print("Pitch : ", error_Pitch.mean(), " ", error_Pitch.std(), " ", error_Pitch.max())
    print("Yaw : ", error_Yaw.mean(), " ", error_Yaw.std(), " ", error_Yaw.max())
    print("X : ", error_X.mean(), " ", error_X.std(), " ", error_X.max())
    print("Y : ", error_Y.mean(), " ", error_Y.std(), " ", error_Y.max())
    print("Z : ", error_Z.mean(), " ", error_Z.std(), " ", error_Z.max())
    

if __name__ == '__main__':
    torch.manual_seed(677)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(677)
    np.random.seed(677)

    torch.multiprocessing.set_start_method('spawn')
    val_oxford_iterative()


 
