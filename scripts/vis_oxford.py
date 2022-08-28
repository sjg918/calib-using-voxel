
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

number_of_frames = 10
log_folder = 'vis_oxford_' + str(number_of_frames)

# choose 0 ~ 99
choosed_frame = 23

# custom miscalib
# [16.5, -13.5, 3.5, 0.65, 0.20, -1.3]
# [4.5, 15.5, -8.5, 0.75, -0.95, 0.40]
custom_miscalib = [4.5, 15.5, -8.5, 0.75, -0.95, 0.40]


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
from CFG.CFG_oxford20 import cfg as cfg
from src.voxelnet import Deepv2_base, Deepv2_2cm
from src.losses import quaternion_distance
from src.oxfordfactory import DataFactory
from src.utils import *

def vis_kitti_iterative():
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
    load_model(model01,cfg.proj_home + 'checkpoint/oxford-1-0.1-(0.5,5).pth', cfg.device)
    
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
    
    pred_Roll_20 = []
    pred_Pitch_20 = []
    pred_Yaw_20 = []
    pred_X_20 = []
    pred_Y_20 = []
    pred_Z_20 = []
    
    pred_Roll_10 = []
    pred_Pitch_10 = []
    pred_Yaw_10 = []
    pred_X_10 = []
    pred_Y_10 = []
    pred_Z_10 = []
    
    pred_Roll_05 = []
    pred_Pitch_05 = []
    pred_Yaw_05 = []
    pred_X_05 = []
    pred_Y_05 = []
    pred_Z_05 = []
    
    pred_Roll_02 = []
    pred_Pitch_02 = []
    pred_Yaw_02 = []
    pred_X_02 = []
    pred_Y_02 = []
    pred_Z_02 = []

    pred_Roll_01 = []
    pred_Pitch_01 = []
    pred_Yaw_01 = []
    pred_X_01 = []
    pred_Y_01 = []
    pred_Z_01 = []
    
    velo_rotx, velo_roty, velo_rotz, velo_trsx, velo_trsy, velo_trsz = custom_miscalib
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

    misRTmat = veloRTmat

    framenumber = predefine_frame_number[choosed_frame].strip()
    framenumber = int(framenumber)
    
    pred_median_na = np.array([0, 0, 0, 0, 0, 0])
    save_bin_file(oxford_dataset20, framenumber, misRTmat, pred_median_na, cfg.proj_home + 'results/' + log_folder + '/input.bin')
    leftnum, rightnum = oxford_dataset20.datalist[framenumber].split(' ')
    rightnum = rightnum.strip()
    right_path = cfg.proj_home + cfg.data_subdir + 'scene2/right/' + rightnum + '.bin'
    rightbin = np.fromfile(right_path, dtype=np.float32).reshape((-1, 3))
    rightbin.tofile(cfg.proj_home + 'results/' + log_folder + '/refer.bin')
    
    gtRTmat = np.linalg.inv(misRTmat)
    gt_Rot = rotationMatrixToEulerAngles(gtRTmat[:3, :3]) * (180.0 / np.pi)
    gt_Trs = gtRTmat[:3, 3]
    pred_median_gt = np.array([gt_Rot[0], gt_Rot[1], gt_Rot[2], gt_Trs[0], gt_Trs[1], gt_Trs[2]])
    save_bin_file(oxford_dataset20, framenumber, misRTmat, pred_median_gt, cfg.proj_home + 'results/' + log_folder + '/gt.bin')

    with torch.no_grad():            
        for cnt in range(framenumber, framenumber+number_of_frames):
            print(cnt)
            leftnum, rightnum = oxford_dataset20.datalist[cnt].split(' ')
            rightnum = rightnum.strip()
            left_path = cfg.proj_home + cfg.data_subdir + 'scene2/left/' + leftnum + '.bin' 
            right_path = cfg.proj_home + cfg.data_subdir + 'scene2/right/' + rightnum + '.bin' 

            leftbin = np.fromfile(left_path, dtype=np.float32).reshape((-1, 3))
            rightbin = np.fromfile(right_path, dtype=np.float32).reshape((-1, 3))

            leftbin = rot_and_trs_points(leftbin, misRTmat)

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
            
            # save(20)
            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]

            pred_Roll_20.append(pred_Rot[0])
            pred_Pitch_20.append(pred_Rot[1])
            pred_Yaw_20.append(pred_Rot[2])
            pred_X_20.append(pred_Trs[0])
            pred_Y_20.append(pred_Trs[1])
            pred_Z_20.append(pred_Trs[2])

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
            
            # save(10)
            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]

            pred_Roll_10.append(pred_Rot[0])
            pred_Pitch_10.append(pred_Rot[1])
            pred_Yaw_10.append(pred_Rot[2])
            pred_X_10.append(pred_Trs[0])
            pred_Y_10.append(pred_Trs[1])
            pred_Z_10.append(pred_Trs[2])

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
            
            # save(05)
            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]

            pred_Roll_05.append(pred_Rot[0])
            pred_Pitch_05.append(pred_Rot[1])
            pred_Yaw_05.append(pred_Rot[2])
            pred_X_05.append(pred_Trs[0])
            pred_Y_05.append(pred_Trs[1])
            pred_Z_05.append(pred_Trs[2])

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
            
            # save(02)
            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]

            pred_Roll_02.append(pred_Rot[0])
            pred_Pitch_02.append(pred_Rot[1])
            pred_Yaw_02.append(pred_Rot[2])
            pred_X_02.append(pred_Trs[0])
            pred_Y_02.append(pred_Trs[1])
            pred_Z_02.append(pred_Trs[2])

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

            # save(01)
            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]

            pred_Roll_01.append(pred_Rot[0])
            pred_Pitch_01.append(pred_Rot[1])
            pred_Yaw_01.append(pred_Rot[2])
            pred_X_01.append(pred_Trs[0])
            pred_Y_01.append(pred_Trs[1])
            pred_Z_01.append(pred_Trs[2])


            continue
        
    pred_median_Roll_20 = np.median(np.array(pred_Roll_20))
    pred_median_Pitch_20 = np.median(np.array(pred_Pitch_20))
    pred_median_Yaw_20 = np.median(np.array(pred_Yaw_20))
    pred_median_X_20 = np.median(np.array(pred_X_20))
    pred_median_Y_20 = np.median(np.array(pred_Y_20))
    pred_median_Z_20 = np.median(np.array(pred_Z_20))
    
    pred_median_20 = np.array([pred_median_Roll_20, pred_median_Pitch_20, pred_median_Yaw_20, pred_median_X_20, pred_median_Y_20, pred_median_Z_20])
    save_bin_file(oxford_dataset20, framenumber, misRTmat, pred_median_20, cfg.proj_home + 'results/' + log_folder + '/pred20.bin')
        
    pred_median_Roll_10 = np.median(np.array(pred_Roll_10))
    pred_median_Pitch_10 = np.median(np.array(pred_Pitch_10))
    pred_median_Yaw_10 = np.median(np.array(pred_Yaw_10))
    pred_median_X_10 = np.median(np.array(pred_X_10))
    pred_median_Y_10 = np.median(np.array(pred_Y_10))
    pred_median_Z_10 = np.median(np.array(pred_Z_10))
    
    pred_median_10 = np.array([pred_median_Roll_10, pred_median_Pitch_10, pred_median_Yaw_10, pred_median_X_10, pred_median_Y_10, pred_median_Z_10])
    save_bin_file(oxford_dataset20, framenumber, misRTmat, pred_median_10, cfg.proj_home + 'results/' + log_folder + '/pred10.bin')
        
    pred_median_Roll_05 = np.median(np.array(pred_Roll_05))
    pred_median_Pitch_05 = np.median(np.array(pred_Pitch_05))
    pred_median_Yaw_05 = np.median(np.array(pred_Yaw_05))
    pred_median_X_05 = np.median(np.array(pred_X_05))
    pred_median_Y_05 = np.median(np.array(pred_Y_05))
    pred_median_Z_05 = np.median(np.array(pred_Z_05))
    
    pred_median_05 = np.array([pred_median_Roll_05, pred_median_Pitch_05, pred_median_Yaw_05, pred_median_X_05, pred_median_Y_05, pred_median_Z_05])
    save_bin_file(oxford_dataset20, framenumber, misRTmat, pred_median_05, cfg.proj_home + 'results/' + log_folder + '/pred05.bin')
        
    pred_median_Roll_02 = np.median(np.array(pred_Roll_02))
    pred_median_Pitch_02 = np.median(np.array(pred_Pitch_02))
    pred_median_Yaw_02 = np.median(np.array(pred_Yaw_02))
    pred_median_X_02 = np.median(np.array(pred_X_02))
    pred_median_Y_02 = np.median(np.array(pred_Y_02))
    pred_median_Z_02 = np.median(np.array(pred_Z_02))
    
    pred_median_02 = np.array([pred_median_Roll_02, pred_median_Pitch_02, pred_median_Yaw_02, pred_median_X_02, pred_median_Y_02, pred_median_Z_02])
    save_bin_file(oxford_dataset20, framenumber, misRTmat, pred_median_02, cfg.proj_home + 'results/' + log_folder + '/pred02.bin')
        
    pred_median_Roll_01 = np.median(np.array(pred_Roll_01))
    pred_median_Pitch_01 = np.median(np.array(pred_Pitch_01))
    pred_median_Yaw_01 = np.median(np.array(pred_Yaw_01))
    pred_median_X_01 = np.median(np.array(pred_X_01))
    pred_median_Y_01 = np.median(np.array(pred_Y_01))
    pred_median_Z_01 = np.median(np.array(pred_Z_01))
    
    pred_median_01 = np.array([pred_median_Roll_01, pred_median_Pitch_01, pred_median_Yaw_01, pred_median_X_01, pred_median_Y_01, pred_median_Z_01])
    save_bin_file(oxford_dataset20, framenumber, misRTmat, pred_median_01, cfg.proj_home + 'results/' + log_folder + '/pred01.bin')
        
    pred01 = np.concatenate((np.array(pred_Roll_01), np.array(pred_Pitch_01), np.array(pred_Yaw_01),
                             np.array(pred_X_01), np.array(pred_Y_01), np.array(pred_Z_01)), axis=0).reshape(6, -1)

    error_Roll = []
    error_Pitch = []
    error_Yaw = []
    error_X = []
    error_Y = []
    error_Z = []

    for number in range(number_of_frames):
        pred_rotx, pred_roty, pred_rotz, pred_trsx, pred_trsy, pred_trsz = pred01[:, number]

        predrotmat = eulerAnglesToRotationMatrix([pred_rotx * (3.141592 / 180.0), pred_roty * (3.141592 / 180.0), pred_rotz * (3.141592 / 180.0)])
        calib_RT_running = np.zeros((4, 4), dtype=np.float32)
        calib_RT_running[:3, :3] = predrotmat[:3,:3]
        calib_RT_running[0, 3] = pred_trsx
        calib_RT_running[1, 3] = pred_trsy
        calib_RT_running[2, 3] = pred_trsz
        calib_RT_running[3, 3] = 1

        # cal error
        error_MAT = calib_RT_running @ misRTmat
        error_ROT = rotationMatrixToEulerAngles(error_MAT[:3, :3]) * (180.0 / np.pi)
        error_TRS = error_MAT[:3, 3]
        error_rot = np.abs(error_ROT)
        error_trs = np.abs(error_TRS)
        
        error_Roll.append(error_rot[0])
        error_Pitch.append(error_rot[1])
        error_Yaw.append(error_rot[2])
        error_X.append(error_trs[0])
        error_Y.append(error_trs[1])
        error_Z.append(error_trs[2])

    error_Roll = np.array(error_Roll)
    error_Pitch = np.array(error_Pitch)
    error_Yaw = np.array(error_Yaw)
    error_X = np.array(error_X)
    error_Y = np.array(error_Y)
    error_Z = np.array(error_Z)
        
    fig, ax1 = plt.subplots(figsize=(4,4), dpi=160)
    ax1.set_xlabel('Rotation Error in °')
    ax1.boxplot([error_Roll, error_Pitch, error_Yaw], showfliers=False)

    plt.ylim([-0.25, 0.25])
    plt.xticks([1,2,3],['Roll', 'Pitch', 'Yaw'])
    plt.savefig(cfg.proj_home +'results/' + log_folder + '/oxford-rotation-boxplot.png', bbox_inches='tight', pad_inches=0)

    fig, ax2 = plt.subplots(figsize=(4,4), dpi=160)
    ax2.set_xlabel('Translation Error in m')
    ax2.boxplot([error_X, error_Y, error_Z], showfliers=False)

    plt.ylim([-0.025, 0.025])
    plt.xticks([1,2,3],['X', 'Y', 'Z'])
    plt.savefig(cfg.proj_home +'results/' + log_folder + '/oxford-trnaslation-boxplot.png', bbox_inches='tight', pad_inches=0)

    print(np.mean(error_Roll), np.mean(error_Pitch), np.mean(error_Yaw), np.mean(error_X), np.mean(error_Y), np.mean(error_Z)) 

     
def save_bin_file(oxford_dataset20, framenumber, misRTmat, pred_median, savefilename):                
    predrotmat = eulerAnglesToRotationMatrix([pred_median[0] * (3.141592 / 180.0), pred_median[1] * (3.141592 / 180.0), pred_median[2] * (3.141592 / 180.0)])
    calib_RT_running = np.zeros((4, 4), dtype=np.float32)
    calib_RT_running[:3, :3] = predrotmat[:3,:3]
    calib_RT_running[0, 3] = pred_median[3]
    calib_RT_running[1, 3] = pred_median[4]
    calib_RT_running[2, 3] = pred_median[5]
    calib_RT_running[3, 3] = 1
    error_MAT = calib_RT_running @ misRTmat
    
    error_ROT = rotationMatrixToEulerAngles(error_MAT[:3, :3]) * (180.0 / np.pi)
    error_TRS = error_MAT[:3, 3]

    error_rot = np.abs(error_ROT)
    error_trs = np.abs(error_TRS)
    np.set_printoptions(precision=4, suppress=True)
    print(error_rot, error_trs)
    
    leftnum, rightnum = oxford_dataset20.datalist[framenumber].split(' ')
    rightnum = rightnum.strip()
    left_path = cfg.proj_home + cfg.data_subdir + 'scene2/left/' + leftnum + '.bin' 
    right_path = cfg.proj_home + cfg.data_subdir + 'scene2/right/' + rightnum + '.bin' 

    leftbin = np.fromfile(left_path, dtype=np.float32).reshape((-1, 3))
    #rightbin = np.fromfile(right_path, dtype=np.float32).reshape((-1, 3))

    leftbin = rot_and_trs_points(leftbin, error_MAT)
    leftbin.tofile(savefilename)


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

# import open3d as o3d
# def vis3dpoints(refpath, tgtpath):
#     refpoints = np.fromfile(refpath, dtype=np.float32).reshape((-1, 3))
#     tgtpoints = np.fromfile(tgtpath, dtype=np.float32).reshape((-1, 3))

#     points = np.concatenate((refpoints, tgtpoints), axis=0)
#     leftbin_colors = np.ones((refpoints.shape[0], 3), dtype=np.uint8)
#     leftbin_colors[:, 2] = 0
#     leftbin_colors[:, 1] = 0
#     rightbin_colors = np.ones((tgtpoints.shape[0], 3), dtype=np.uint8)
#     rightbin_colors[:, 0] = 0
#     rightbin_colors[:, 2] = 0
#     np_colors = np.concatenate((leftbin_colors, rightbin_colors), axis=0)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(np_colors)
#     o3d.visualization.draw_geometries([pcd])
    
if __name__ == '__main__':
    torch.manual_seed(677)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(677)
    np.random.seed(677)

    torch.multiprocessing.set_start_method('spawn')
    vis_kitti_iterative()

    #vis3dpoints(cfg.proj_home + 'results/' + log_folder + '/refer.bin', cfg.proj_home + 'results/' + log_folder + '/input.bin')
    #vis3dpoints(cfg.proj_home + 'results/' + log_folder + '/refer.bin', cfg.proj_home + 'results/' + log_folder + '/pred01.bin')
