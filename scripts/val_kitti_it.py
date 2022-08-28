
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from CFG.CFG_kitti20 import cfg as cfg20
from CFG.CFG_kitti1 import cfg as cfg2cm
from src.voxelnet import Deepv2_base, Deepv2_2cm
from src.losses import quaternion_distance
from src.kittifactory import DataFactory
from src.sgmgpu.utils import mkDispmap
from src.utils import *

def vis_kitti_iterative():
    #print(cfg20.model)
    
    model20 = Deepv2_base(cfg20.sgm_local, cfg20.velo_local).to(cfg20.device)
    model10 = Deepv2_base(cfg20.sgm_local, cfg20.velo_local).to(cfg20.device)
    model05 = Deepv2_base(cfg20.sgm_local, cfg20.velo_local).to(cfg20.device)
    model02 = Deepv2_base(cfg20.sgm_local, cfg20.velo_local).to(cfg20.device)
    model01 = Deepv2_2cm(cfg2cm.sgm_local, cfg2cm.velo_local).to(cfg20.device)

    load_model(model20, cfg20.proj_home + 'checkpoint/kitti-20.pth', cfg20.device)
    load_model(model10, cfg20.proj_home + 'checkpoint/kitti-10.pth', cfg20.device)
    load_model(model05, cfg20.proj_home + 'checkpoint/kitti-5.pth', cfg20.device)
    load_model(model02, cfg20.proj_home + 'checkpoint/kitti-2.pth', cfg20.device)
    load_model(model01, cfg20.proj_home + 'checkpoint/kitti-1-0.1-vx2.5-(0.5,5).pth', cfg2cm.device)

    point2voxel20 = PointToVoxel(vsize_xyz=cfg20.voxelsize,
                           coors_range_xyz=cfg20.voxelrange,
                           num_point_features=4,
                           max_num_voxels=cfg20.maxvoxels,
                           max_num_points_per_voxel=cfg20.maxpoints,
                           device=cfg20.device)
    
    point2voxel2cm = PointToVoxel(vsize_xyz=cfg2cm.voxelsize,
                           coors_range_xyz=cfg2cm.voxelrange,
                           num_point_features=4,
                           max_num_voxels=cfg2cm.maxvoxels,
                           max_num_points_per_voxel=cfg2cm.maxpoints,
                           device=cfg20.device)

    # define dataloader
    kitti_dataset20 = DataFactory(cfg20, 'val')

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
        for cnt in range(len(kitti_dataset20)):
            print(cnt, len(kitti_dataset20), end='\r')
            sgmroad, veloroad, tgtRTmat = kitti_dataset20[cnt]
            misRTmat = np.linalg.inv(tgtRTmat)

            # forward(20)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, veloroad, point2voxel20, cfg20)
            quaternion, translate = model20(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu().numpy()
            
            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, misRTmat)            
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_20.append(error_rot[2])
            error_Pitch_20.append(error_rot[0])
            error_Yaw_20.append(error_rot[1])
            error_X_20.append(error_trs[0])
            error_Y_20.append(error_trs[1])
            error_Z_20.append(error_trs[2])
            
            # calib
            run_calib_RT = np.zeros((4,4), dtype=np.float32)
            run_calib_RT[:3, :3] = rot_matrix
            run_calib_RT[0, 3] = translate[0]
            run_calib_RT[1, 3] = translate[1]
            run_calib_RT[2, 3] = translate[2]
            run_calib_RT[3, 3] = 1
            newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

            # forward(10)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
            quaternion, translate = model10(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu()
            
            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, run_calib_RT @ misRTmat)
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_10.append(error_rot[2])
            error_Pitch_10.append(error_rot[0])
            error_Yaw_10.append(error_rot[1])
            error_X_10.append(error_trs[0])
            error_Y_10.append(error_trs[1])
            error_Z_10.append(error_trs[2])

            # calib (10)
            calib_RT = np.zeros((4,4), dtype=np.float32)
            calib_RT[:3, :3] = rot_matrix
            calib_RT[0, 3] = translate[0]
            calib_RT[1, 3] = translate[1]
            calib_RT[2, 3] = translate[2]
            calib_RT[3, 3] = 1
            run_calib_RT = calib_RT @ run_calib_RT
            newveloroad = rot_and_trs_points(veloroad, run_calib_RT)
            
            # forward(05)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
            quaternion, translate = model05(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu()
            
            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, run_calib_RT @ misRTmat)
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_05.append(error_rot[2])
            error_Pitch_05.append(error_rot[0])
            error_Yaw_05.append(error_rot[1])
            error_X_05.append(error_trs[0])
            error_Y_05.append(error_trs[1])
            error_Z_05.append(error_trs[2])
            
            # calib (02)
            calib_RT = np.zeros((4,4), dtype=np.float32)
            calib_RT[:3, :3] = rot_matrix
            calib_RT[0, 3] = translate[0]
            calib_RT[1, 3] = translate[1]
            calib_RT[2, 3] = translate[2]
            calib_RT[3, 3] = 1
            run_calib_RT = calib_RT @ run_calib_RT
            newveloroad = rot_and_trs_points(veloroad, run_calib_RT)
            
            # forward(02)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
            quaternion, translate = model02(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu()
            
            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, run_calib_RT @ misRTmat)
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_02.append(error_rot[2])
            error_Pitch_02.append(error_rot[0])
            error_Yaw_02.append(error_rot[1])
            error_X_02.append(error_trs[0])
            error_Y_02.append(error_trs[1])
            error_Z_02.append(error_trs[2])
            
            # calib (01)
            calib_RT = np.zeros((4,4), dtype=np.float32)
            calib_RT[:3, :3] = rot_matrix
            calib_RT[0, 3] = translate[0]
            calib_RT[1, 3] = translate[1]
            calib_RT[2, 3] = translate[2]
            calib_RT[3, 3] = 1
            run_calib_RT = calib_RT @ run_calib_RT
            newveloroad = rot_and_trs_points(veloroad, run_calib_RT)
            
            # forward(01)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel2cm, cfg2cm)
            quaternion, translate = model01(sgm_input, velo_input)
            quaternion = quaternion.squeeze(dim=0).cpu().numpy()
            rot_matrix = rotation_matrix_from_quaternion(quaternion)
            translate = translate.squeeze(dim=0).cpu()
            
            # cal error
            error_ROT, error_TRS = cal_error(rot_matrix, translate, run_calib_RT @ misRTmat)
            error_rot = np.abs(error_ROT)
            error_trs = np.abs(error_TRS)
            error_Roll_01.append(error_rot[2])
            error_Pitch_01.append(error_rot[0])
            error_Yaw_01.append(error_rot[1])
            error_X_01.append(error_trs[0])
            error_Y_01.append(error_trs[1])
            error_Z_01.append(error_trs[2])
            
            #print(cnt, len(kitti_dataset20), error_rot, error_trs)

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
    vis_kitti_iterative()


 
