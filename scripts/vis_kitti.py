
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

number_of_frames = 100
log_folder = 'vis_kitti_' + str(number_of_frames)

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
from CFG.CFG_kitti20 import cfg as cfg20
from CFG.CFG_kitti1 import cfg as cfg2cm
from src.voxelnet import Deepv2_base, Deepv2_2cm
from src.losses import quaternion_distance
from src.kittifactory import DataFactory
from src.sgmgpu.utils import mkDispmap
from src.utils import *

def vis_kitti_iterative():
    if os.path.exists(cfg20.proj_home + 'results/' + log_folder + '/'):
        shutil.rmtree(cfg20.proj_home + 'results/' + log_folder + '/')
    os.makedirs(cfg20.proj_home + 'results/' + log_folder + '/')
    
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
    
    with open(cfg20.proj_home + 'gendata/100miscalib.txt', 'r') as f:
        predefine_error_list = f.readlines()
        
    with open(cfg20.proj_home + 'gendata/100number-kitti.txt', 'r') as f:
        predefine_frame_number = f.readlines()

    # define dataloader
    kitti_dataset20 = DataFactory(cfg20, 'val')

    model20.eval()
    model10.eval()
    model05.eval()
    model02.eval()
    model01.eval()
    
    pred_rotx_20 = []
    pred_roty_20 = []
    pred_rotz_20 = []
    pred_trsx_20 = []
    pred_trsy_20 = []
    pred_trsz_20 = []
    
    pred_rotx_10 = []
    pred_roty_10 = []
    pred_rotz_10 = []
    pred_trsx_10 = []
    pred_trsy_10 = []
    pred_trsz_10 = []
    
    pred_rotx_05 = []
    pred_roty_05 = []
    pred_rotz_05 = []
    pred_trsx_05 = []
    pred_trsy_05 = []
    pred_trsz_05 = []
    
    pred_rotx_02 = []
    pred_roty_02 = []
    pred_rotz_02 = []
    pred_trsx_02 = []
    pred_trsy_02 = []
    pred_trsz_02 = []

    pred_rotx_01 = []
    pred_roty_01 = []
    pred_rotz_01 = []
    pred_trsx_01 = []
    pred_trsy_01 = []
    pred_trsz_01 = []
    
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

    with torch.no_grad():            
        for cnt in range(framenumber, framenumber+number_of_frames):
            print(cnt)
            seq, imgnum = kitti_dataset20.datalist[cnt].split(' ')
            imgnum = imgnum.strip()
            sgm_path = cfg20.proj_home + cfg20.data_subdir + seq + '/disp_' + str(cfg20.sgm_path) + '/' + imgnum + '.bin'
            velo_path = cfg20.proj_home + cfg20.data_subdir + seq + '/velodyne_' + str(cfg20.lidar_channel) + '/' + imgnum + '.bin'
            calib_path = cfg20.odometry_home + cfg20.calib_subdir + seq + '/calib.txt'

            filedata = read_calib_file(calib_path)
            P_rect_20 = np.reshape(filedata['P2'], (3, 4))
            T_cam0_velo = np.reshape(filedata['Tr'], (3, 4))
            T_cam0_velo = np.vstack([T_cam0_velo, [0, 0, 0, 1]])
            T2 = np.eye(4)
            T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
            initcalib_RT = T2.dot(T_cam0_velo)
            
            sgmroad = np.fromfile(sgm_path, dtype=np.float32).reshape((-1, 3))
            veloroad = np.fromfile(velo_path, dtype=np.float32).reshape((-1, 3))
            veloroad = rot_and_trs_points(veloroad, initcalib_RT)
            veloroad = rot_and_trs_points(veloroad, misRTmat)

            # forward(20)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, veloroad, point2voxel20, cfg20)
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
            newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]
            pred_rotx_20.append(pred_Rot[0])
            pred_roty_20.append(pred_Rot[1])
            pred_rotz_20.append(pred_Rot[2])
            pred_trsx_20.append(pred_Trs[0])
            pred_trsy_20.append(pred_Trs[1])
            pred_trsz_20.append(pred_Trs[2])

            # forward(10)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
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

            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]
            pred_rotx_10.append(pred_Rot[0])
            pred_roty_10.append(pred_Rot[1])
            pred_rotz_10.append(pred_Rot[2])
            pred_trsx_10.append(pred_Trs[0])
            pred_trsy_10.append(pred_Trs[1])
            pred_trsz_10.append(pred_Trs[2])

            newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

            # forward(05)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
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

            #            
            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]
            pred_rotx_05.append(pred_Rot[0])
            pred_roty_05.append(pred_Rot[1])
            pred_rotz_05.append(pred_Rot[2])
            pred_trsx_05.append(pred_Trs[0])
            pred_trsy_05.append(pred_Trs[1])
            pred_trsz_05.append(pred_Trs[2])

            newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

            # forward(02)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
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

            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]
            pred_rotx_02.append(pred_Rot[0])
            pred_roty_02.append(pred_Rot[1])
            pred_rotz_02.append(pred_Rot[2])
            pred_trsx_02.append(pred_Trs[0])
            pred_trsy_02.append(pred_Trs[1])
            pred_trsz_02.append(pred_Trs[2])

            newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

            # forward(01)
            sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel2cm, cfg2cm)
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
            
            #
            pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / np.pi)
            pred_Trs = run_calib_RT[:3, 3]
            pred_rotx_01.append(pred_Rot[0])
            pred_roty_01.append(pred_Rot[1])
            pred_rotz_01.append(pred_Rot[2])
            pred_trsx_01.append(pred_Trs[0])
            pred_trsy_01.append(pred_Trs[1])
            pred_trsz_01.append(pred_Trs[2])


            continue
        
    pred_median_na = np.array([0, 0, 0, 0, 0, 0])
    save_projection_image(kitti_dataset20, framenumber, misRTmat, pred_median_na, cfg20.proj_home + 'results/' + log_folder + '/input.png')
    
    gtRTmat = np.linalg.inv(misRTmat)
    gt_Rot = rotationMatrixToEulerAngles(gtRTmat[:3, :3]) * (180.0 / np.pi)
    gt_Trs = gtRTmat[:3, 3]
    pred_median_gt = np.array([gt_Rot[0], gt_Rot[1], gt_Rot[2], gt_Trs[0], gt_Trs[1], gt_Trs[2]])
    save_projection_image(kitti_dataset20, framenumber, misRTmat, pred_median_gt, cfg20.proj_home + 'results/' + log_folder + '/gt.png')
        
    pred_median_rotx_20 = np.median(np.array(pred_rotx_20))
    pred_median_roty_20 = np.median(np.array(pred_roty_20))
    pred_median_rotz_20 = np.median(np.array(pred_rotz_20))
    pred_median_trsx_20 = np.median(np.array(pred_trsx_20))
    pred_median_trsy_20 = np.median(np.array(pred_trsy_20))
    pred_median_trsz_20 = np.median(np.array(pred_trsz_20))
    
    pred_median_20 = np.array([pred_median_rotx_20, pred_median_roty_20, pred_median_rotz_20, pred_median_trsx_20, pred_median_trsy_20, pred_median_trsz_20])
    save_projection_image(kitti_dataset20, framenumber, misRTmat, pred_median_20, cfg20.proj_home + 'results/' + log_folder + '/pred20.png')
        
    pred_median_rotx_10 = np.median(np.array(pred_rotx_10))
    pred_median_roty_10 = np.median(np.array(pred_roty_10))
    pred_median_rotz_10 = np.median(np.array(pred_rotz_10))
    pred_median_trsx_10 = np.median(np.array(pred_trsx_10))
    pred_median_trsy_10 = np.median(np.array(pred_trsy_10))
    pred_median_trsz_10 = np.median(np.array(pred_trsz_10))
    
    pred_median_10 = np.array([pred_median_rotx_10, pred_median_roty_10, pred_median_rotz_10, pred_median_trsx_10, pred_median_trsy_10, pred_median_trsz_10])
    save_projection_image(kitti_dataset20, framenumber, misRTmat, pred_median_10, cfg20.proj_home + 'results/' + log_folder + '/pred10.png')
        
    pred_median_rotx_05 = np.median(np.array(pred_rotx_05))
    pred_median_roty_05 = np.median(np.array(pred_roty_05))
    pred_median_rotz_05 = np.median(np.array(pred_rotz_05))
    pred_median_trsx_05 = np.median(np.array(pred_trsx_05))
    pred_median_trsy_05 = np.median(np.array(pred_trsy_05))
    pred_median_trsz_05 = np.median(np.array(pred_trsz_05))
    
    pred_median_05 = np.array([pred_median_rotx_05, pred_median_roty_05, pred_median_rotz_05, pred_median_trsx_05, pred_median_trsy_05, pred_median_trsz_05])
    save_projection_image(kitti_dataset20, framenumber, misRTmat, pred_median_05, cfg20.proj_home + 'results/' + log_folder + '/pred05.png')
        
    pred_median_rotx_02 = np.median(np.array(pred_rotx_02))
    pred_median_roty_02 = np.median(np.array(pred_roty_02))
    pred_median_rotz_02 = np.median(np.array(pred_rotz_02))
    pred_median_trsx_02 = np.median(np.array(pred_trsx_02))
    pred_median_trsy_02 = np.median(np.array(pred_trsy_02))
    pred_median_trsz_02 = np.median(np.array(pred_trsz_02))
    
    pred_median_02 = np.array([pred_median_rotx_02, pred_median_roty_02, pred_median_rotz_02, pred_median_trsx_02, pred_median_trsy_02, pred_median_trsz_02])
    save_projection_image(kitti_dataset20, framenumber, misRTmat, pred_median_02, cfg20.proj_home + 'results/' + log_folder + '/pred02.png')
        
    pred_median_rotx_01 = np.median(np.array(pred_rotx_01))
    pred_median_roty_01 = np.median(np.array(pred_roty_01))
    pred_median_rotz_01 = np.median(np.array(pred_rotz_01))
    pred_median_trsx_01 = np.median(np.array(pred_trsx_01))
    pred_median_trsy_01 = np.median(np.array(pred_trsy_01))
    pred_median_trsz_01 = np.median(np.array(pred_trsz_01))
    
    pred_median_01 = np.array([pred_median_rotx_01, pred_median_roty_01, pred_median_rotz_01, pred_median_trsx_01, pred_median_trsy_01, pred_median_trsz_01])
    save_projection_image(kitti_dataset20, framenumber, misRTmat, pred_median_01, cfg20.proj_home + 'results/' + log_folder + '/pred01.png')
    
    pred01 = np.concatenate((np.array(pred_rotx_01), np.array(pred_roty_01), np.array(pred_rotz_01),
                             np.array(pred_trsx_01), np.array(pred_trsy_01), np.array(pred_trsz_01)), axis=0).reshape(6, -1)

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
        
        error_Roll.append(error_rot[2])
        error_Pitch.append(error_rot[0])
        error_Yaw.append(error_rot[1])
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
    plt.savefig(cfg20.proj_home + 'results/' + log_folder + '/kitti-rotation-boxplot.png', bbox_inches='tight', pad_inches=0)

    fig, ax2 = plt.subplots(figsize=(4,4), dpi=160)
    ax2.set_xlabel('Translation Error in m')
    ax2.boxplot([error_X, error_Y, error_Z], showfliers=False)

    plt.ylim([-0.025, 0.025])
    plt.xticks([1,2,3],['X', 'Y', 'Z'])
    plt.savefig(cfg20.proj_home + 'results/' + log_folder + '/kitti-trnaslation-boxplot.png', bbox_inches='tight', pad_inches=0)

    print(np.mean(error_Roll), np.mean(error_Pitch), np.mean(error_Yaw), np.mean(error_X), np.mean(error_Y), np.mean(error_Z)) 
     

def save_projection_image(kitti_dataset20, framenumber, tgtRTmat, pred_median, savefilename):                
    predrotmat = eulerAnglesToRotationMatrix([pred_median[0] * (3.141592 / 180.0), pred_median[1] * (3.141592 / 180.0), pred_median[2] * (3.141592 / 180.0)])
    calib_RT_running = np.zeros((4, 4), dtype=np.float32)
    calib_RT_running[:3, :3] = predrotmat[:3,:3]
    calib_RT_running[0, 3] = pred_median[3]
    calib_RT_running[1, 3] = pred_median[4]
    calib_RT_running[2, 3] = pred_median[5]
    calib_RT_running[3, 3] = 1
    error_MAT = calib_RT_running @ tgtRTmat
    
    #error_ROT = rotationMatrixToEulerAngles(error_MAT[:3, :3]) * (180.0 / np.pi)
    #error_TRS = error_MAT[:3, 3]
                          
    seq, imgnum = kitti_dataset20.datalist[framenumber].split(' ')
    imgnum = imgnum.strip()
    left_image_path = cfg20.odometry_home + 'data_odometry_color/dataset/sequences/' + seq + '/image_2/' + imgnum + '.png'
    left_image = cv2.imread(left_image_path)
    h, w, c = left_image.shape
    calib_path = cfg20.odometry_home + cfg20.calib_subdir + seq + '/calib.txt'
    odom = odometry(cfg20.odometry_home + 'data_odometry_color/dataset/', seq)
    calib = odom.calib
    K = calib.K_cam2
    filedata = read_calib_file(calib_path)
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    T_cam0_velo = np.reshape(filedata['Tr'], (3, 4))
    T_cam0_velo = np.vstack([T_cam0_velo, [0, 0, 0, 1]])
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    initcalib_RT = T2.dot(T_cam0_velo)
                          
    velo_path = cfg20.proj_home + cfg20.data_subdir + seq + '/velodyne_' + str(cfg20.lidar_channel) + '/' + imgnum + '.bin'
    veloroad = np.fromfile(velo_path, dtype=np.float32).reshape((-1, 3))
    veloroad = rot_and_trs_points(veloroad, initcalib_RT)
    veloroad = rot_and_trs_points(veloroad, error_MAT)
    
    c_u = K[0, 2]
    c_v = K[1, 2]
    f_u = K[0, 0]
    f_v = K[1, 1]
    baseline = 0.54
    
    u = f_u * veloroad[:, 0] / veloroad[:, 2] + c_u
    v = f_v * veloroad[:, 1] / veloroad[:, 2] + c_v
    z = veloroad[:, 2] * 4.5
    
    mask = (u >= 0) * (u < w) * (v >= 0) * (v < h) * (z > 0)
    u = u[mask]
    v = v[mask]
    z = z[mask]
    
    u = np.clip(u.astype(np.int32), 0, w-1)
    v = np.clip(v.astype(np.int32), 0, h-1)
    z = cv2.applyColorMap(z.astype(np.uint8), cv2.COLORMAP_TURBO)
        
    for i in range(u.shape[0]):
        c1 = int(z[i, 0, 0])
        c2 = int(z[i, 0, 1])
        c3 = int(z[i, 0, 2])
        cv2.circle(left_image, (u[i], v[i]), 1, (c1,c2,c3), -1)
                
    cv2.imwrite(savefilename, left_image)


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


if __name__ == '__main__':
    torch.manual_seed(677)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(677)
    np.random.seed(677)

    torch.multiprocessing.set_start_method('spawn')
    vis_kitti_iterative()


 
