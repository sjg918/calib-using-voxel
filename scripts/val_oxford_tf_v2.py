
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

number_of_frames = 10
log_folder = 'val_oxford_tf_v2_' + str(number_of_frames)

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
import matplotlib.pyplot as pl

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.CFG_oxford20 import cfg as cfg
from src.voxelnet import Deepv2_base
from src.losses import quaternion_distance
from src.oxfordfactory import DataFactory
from src.utils import *

def val_oxford_temporal_filtering():
    if os.path.exists(cfg.proj_home + 'results/' + log_folder + '/'):
        shutil.rmtree(cfg.proj_home + 'results/' + log_folder + '/')
    os.makedirs(cfg.proj_home + 'results/' + log_folder + '/')
    
    model20 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    model10 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    model05 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    model02 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    model01 = Deepv2_base(cfg.left_local, cfg.right_local, split=False).to(cfg.device)
    
    load_model(model20,cfg.proj_home + 'checkpoint/oxford-20.pth', cfg.device)
    load_model(model10,cfg.proj_home + 'checkpoint/oxford-10.pth', cfg.device)
    load_model(model05,cfg.proj_home + 'checkpoint/oxford-5.pth', cfg.device)
    load_model(model02,cfg.proj_home + 'checkpoint/oxford-2.pth', cfg.device)
    load_model(model01,cfg.proj_home + 'checkpoint/oxford-1.pth', cfg.device)

    point2voxel20 = PointToVoxel(vsize_xyz=cfg.voxelsize,
                           coors_range_xyz=cfg.voxelrange,
                           num_point_features=4,
                           max_num_voxels=cfg.maxvoxels,
                           max_num_points_per_voxel=cfg.maxpoints,
                           device=cfg.device)

    with open(cfg.proj_home + 'gendata/100miscalib.txt', 'r') as f:
        predefine_error_list = f.readlines()
        
    with open(cfg.proj_home + 'gendata/100number-oxford.txt', 'r') as f:
        predefine_frame_number = f.readlines()

    # define dataloader
    oxford_dataset20 = DataFactory(cfg, 'val')

    model20.eval()
    model10.eval()
    model05.eval()
    model02.eval()
    model01.eval()
    
    with torch.no_grad():
        for cnt_ in range(100):
            
            pred_Roll_01 = []
            pred_Pitch_01 = []
            pred_Yaw_01 = []
            pred_X_01 = []
            pred_Y_01 = []
            pred_Z_01 = []
            
            print(cnt_)

            velo_rotx, velo_roty, velo_rotz, velo_trsx, velo_trsy, velo_trsz = predefine_error_list[cnt_].strip().split(' ')
            velo_rotx = float(velo_rotx) * (3.141592 / 180.0)
            velo_roty = float(velo_roty) * (3.141592 / 180.0)
            velo_rotz = float(velo_rotz) * (3.141592 / 180.0)
            velo_trsx = float(velo_trsx)
            velo_trsy = float(velo_trsy)
            velo_trsz = float(velo_trsz)

            velorotmat = eulerAnglesToRotationMatrix([velo_rotx, velo_roty, velo_rotz])
            veloRTmat = np.zeros((4, 4), dtype=np.float32)
            veloRTmat[:3, :3] = velorotmat[:3,:3]
            veloRTmat[0, 3] = velo_trsx
            veloRTmat[1, 3] = velo_trsy
            veloRTmat[2, 3] = velo_trsz
            veloRTmat[3, 3] = 1

            tgtRTmat = veloRTmat
            
            framenumber = predefine_frame_number[cnt_].strip()
            framenumber = int(framenumber)
            
                            
            for cnt in range(framenumber, framenumber+number_of_frames):
                leftnum, rightnum = oxford_dataset20.datalist[cnt].split(' ')
                rightnum = rightnum.strip()
                left_path = cfg.proj_home + cfg.data_subdir + 'scene2/left/' + leftnum + '.bin' 
                right_path = cfg.proj_home + cfg.data_subdir + 'scene2/right/' + rightnum + '.bin' 
                
                leftbin = np.fromfile(left_path, dtype=np.float32).reshape((-1, 3))
                rightbin = np.fromfile(right_path, dtype=np.float32).reshape((-1, 3))
                
                leftbin = rot_and_trs_points(leftbin, tgtRTmat)

                # forward(20)
                sgm_input, velo_input = mk_sparsely_embedded_tensor(rightbin, leftbin, point2voxel20, cfg)
                quaternion, translate = model20(sgm_input, velo_input)
                quaternion = quaternion.squeeze(dim=0).cpu().numpy()
                rot_matrix = rotation_matrix_from_quaternion(quaternion)
                translate = translate.squeeze(dim=0).cpu().numpy()
                
                # calib(20)
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
                
                # calib(10)
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
                
                # calib(05)
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
                
                # calib(02)
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
                
                # calib(01)
                calib_RT = np.zeros((4,4), dtype=np.float32)
                calib_RT[:3, :3] = rot_matrix
                calib_RT[0, 3] = translate[0]
                calib_RT[1, 3] = translate[1]
                calib_RT[2, 3] = translate[2]
                calib_RT[3, 3] = 1
                run_calib_RT = calib_RT @ run_calib_RT
                
                pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
                pred_Trs = run_calib_RT[:3, 3]
                
                pred_Roll_01.append(pred_Rot[0])
                pred_Pitch_01.append(pred_Rot[1])
                pred_Yaw_01.append(pred_Rot[2])
                pred_X_01.append(pred_Trs[0])
                pred_Y_01.append(pred_Trs[1])
                pred_Z_01.append(pred_Trs[2])
                
                
                # cal error
                #error_MAT = calib_RT_running @ tgtRTmat
                #error_ROT = rotationMatrixToEulerAngles(error_MAT[:3, :3]) * (180.0 / np.pi)
                #error_TRS = error_MAT[:3, 3]

                #error_rot = np.abs(error_ROT)
                #error_trs = np.abs(error_TRS)
                
                #print(error_rot, error_trs)
                #df=df
                
                
                continue
                
            pred_median_Roll = np.median(np.array(pred_Roll_01))
            pred_median_Pitch = np.median(np.array(pred_Pitch_01))
            pred_median_Yaw = np.median(np.array(pred_Yaw_01))
            pred_median_X = np.median(np.array(pred_X_01))
            pred_median_Y = np.median(np.array(pred_Y_01))
            pred_median_Z = np.median(np.array(pred_Z_01))
            
            pred_median = np.array([pred_median_Roll, pred_median_Pitch, pred_median_Yaw, pred_median_X, pred_median_Y, pred_median_Z])
            np.save(cfg.proj_home + 'results/' + log_folder + '/pred_median_' + str(cnt_) +'.npy', pred_median)

            continue
    
    
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


if __name__ == '__main__':
    torch.manual_seed(677)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(677)
    np.random.seed(677)
    
    torch.multiprocessing.set_start_method('spawn')
    val_oxford_temporal_filtering()
 
    eval_100miscalib(cfg.proj_home + 'results/' + log_folder + '/', cfg.proj_home + 'gendata/100miscalib.txt', kitti=False)
