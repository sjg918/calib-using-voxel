
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

number_of_frames = 10
log_folder = 'val_kitti_tf_v2_' + str(number_of_frames)

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

def val_kitti_temporal_filtering():
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
    
    error_rotx = []
    error_roty = []
    error_rotz = []
    error_trsx = []
    error_trsy = []
    error_trsz = []

    timelist = []
    
    with torch.no_grad():
        for cnt_ in range(100):
            
            pred_rotx = []
            pred_roty = []
            pred_rotz = []
            pred_trsx = []
            pred_trsy = []
            pred_trsz = []
            
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

            misRTmat = veloRTmat
            
            framenumber = predefine_frame_number[cnt_].strip()
            framenumber = int(framenumber)
                            
            for cnt in range(framenumber, framenumber + number_of_frames):
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
                time20_start = time.time()
                sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, veloroad, point2voxel20, cfg20)
                quaternion, translate = model20(sgm_input, velo_input)
                quaternion = quaternion.squeeze(dim=0).cpu().numpy()
                rot_matrix = rotation_matrix_from_quaternion(quaternion)
                translate = translate.squeeze(dim=0).cpu().numpy()
                time20_end = time.time()

                # calib(20)
                calib_RT = np.zeros((4,4), dtype=np.float32)
                calib_RT[:3, :3] = rot_matrix
                calib_RT[0, 3] = translate[0]
                calib_RT[1, 3] = translate[1]
                calib_RT[2, 3] = translate[2]
                calib_RT[3, 3] = 1
                run_calib_RT = calib_RT
                newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

                # forward(10)
                time10_start = time.time()
                sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
                quaternion, translate = model10(sgm_input, velo_input)
                quaternion = quaternion.squeeze(dim=0).cpu().numpy()
                rot_matrix = rotation_matrix_from_quaternion(quaternion)
                translate = translate.squeeze(dim=0).cpu()
                time10_end = time.time()
                
                # calib(10)
                calib_RT = np.zeros((4,4), dtype=np.float32)
                calib_RT[:3, :3] = rot_matrix
                calib_RT[0, 3] = translate[0]
                calib_RT[1, 3] = translate[1]
                calib_RT[2, 3] = translate[2]
                calib_RT[3, 3] = 1
                run_calib_RT = calib_RT @ run_calib_RT
                newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

                # forward(05)
                time5_start = time.time()
                sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
                quaternion, translate = model05(sgm_input, velo_input)
                quaternion = quaternion.squeeze(dim=0).cpu().numpy()
                rot_matrix = rotation_matrix_from_quaternion(quaternion)
                translate = translate.squeeze(dim=0).cpu()
                time5_end = time.time()
                
                # calib(05)
                calib_RT = np.zeros((4,4), dtype=np.float32)
                calib_RT[:3, :3] = rot_matrix
                calib_RT[0, 3] = translate[0]
                calib_RT[1, 3] = translate[1]
                calib_RT[2, 3] = translate[2]
                calib_RT[3, 3] = 1
                run_calib_RT = calib_RT @ run_calib_RT
                newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

                # forward(02)
                time2_start = time.time()
                sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel20, cfg20)
                quaternion, translate = model02(sgm_input, velo_input)
                quaternion = quaternion.squeeze(dim=0).cpu().numpy()
                rot_matrix = rotation_matrix_from_quaternion(quaternion)
                translate = translate.squeeze(dim=0).cpu()
                time2_end = time.time()
                
                # calib(02)
                calib_RT = np.zeros((4,4), dtype=np.float32)
                calib_RT[:3, :3] = rot_matrix
                calib_RT[0, 3] = translate[0]
                calib_RT[1, 3] = translate[1]
                calib_RT[2, 3] = translate[2]
                calib_RT[3, 3] = 1
                run_calib_RT = calib_RT @ run_calib_RT
                newveloroad = rot_and_trs_points(veloroad, run_calib_RT)

                # forward(01)
                time1_start = time.time()
                sgm_input, velo_input = mk_sparsely_embedded_tensor(sgmroad, newveloroad, point2voxel2cm, cfg2cm)
                quaternion, translate = model01(sgm_input, velo_input)
                quaternion = quaternion.squeeze(dim=0).cpu().numpy()
                rot_matrix = rotation_matrix_from_quaternion(quaternion)
                translate = translate.squeeze(dim=0).cpu()
                time1_end = time.time()
                
                # calib(01)
                calib_RT = np.zeros((4,4), dtype=np.float32)
                calib_RT[:3, :3] = rot_matrix
                calib_RT[0, 3] = translate[0]
                calib_RT[1, 3] = translate[1]
                calib_RT[2, 3] = translate[2]
                calib_RT[3, 3] = 1
                run_calib_RT = calib_RT @ run_calib_RT
                
                #
                pred_Rot = rotationMatrixToEulerAngles(run_calib_RT[:3, :3]) * (180.0 / 3.141592)
                pred_Trs = run_calib_RT[:3, 3]
                
                pred_rotx.append(pred_Rot[0])
                pred_roty.append(pred_Rot[1])
                pred_rotz.append(pred_Rot[2])
                pred_trsx.append(pred_Trs[0])
                pred_trsy.append(pred_Trs[1])
                pred_trsz.append(pred_Trs[2])
                
                if cnt_ == 0 and cnt == 0:
                    pass
                else:
                    timelist.append(
                        [time20_end - time20_start,
                        time10_end - time10_start,
                        time5_end - time5_start,
                        time2_end - time2_start,
                        time1_end - time1_start]
                    ) 
                continue
                
            pred_median_rotx = np.median(np.array(pred_rotx))
            pred_median_roty = np.median(np.array(pred_roty))
            pred_median_rotz = np.median(np.array(pred_rotz))
            pred_median_trsx = np.median(np.array(pred_trsx))
            pred_median_trsy = np.median(np.array(pred_trsy))
            pred_median_trsz = np.median(np.array(pred_trsz))

            pred_median = np.array([pred_median_rotx, pred_median_roty, pred_median_rotz, pred_median_trsx, pred_median_trsy, pred_median_trsz])
            np.save(cfg20.proj_home + 'results/' + log_folder + '/pred_median_' + str(cnt_) +'.npy', pred_median)
            
            continue
            
    error_rotx = np.array(error_rotx)
    error_roty = np.array(error_roty)
    error_rotz = np.array(error_rotz)
    error_trsx = np.array(error_trsx)
    error_trsy = np.array(error_trsy)
    error_trsz = np.array(error_trsz)
    timelist = np.array(timelist)
    
    print('meantime')
    print(timelist[:, 0].mean())
    print(timelist[:, 1].mean())
    print(timelist[:, 2].mean())
    print(timelist[:, 3].mean())
    print(timelist[:, 4].mean())


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
    #val_kitti_temporal_filtering()

    print(cfg20.proj_home + 'results/' + log_folder + '/')
    eval_100miscalib(cfg20.proj_home + 'results/' + log_folder + '/', cfg20.proj_home + 'gendata/100miscalib.txt')
 
