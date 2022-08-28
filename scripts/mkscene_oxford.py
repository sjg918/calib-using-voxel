
import numpy as np
import cv2
import os
import shutil
import time
import sys
import math
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.CFG_oxford20 import cfg
import torch


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.
    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.
    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix
    Raises:
        ValueError: if `len(rpy) != 3`.
    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                     [0, math.sin(rpy[0]), math.cos(rpy[0])]])
    R_y = np.matrix([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                     [0, 1, 0],
                     [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
    R_z = np.matrix([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                     [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx


leftlidar2cam = [-0.60072, -0.34077, -0.26837, -0.0053948, -0.041998, -3.1337]
rightlidar2cam = [-0.61153, 0.55676, -0.27023, 0.0027052, -0.041999, -3.1357]

leftmatrixR = np.zeros((4,4), dtype=np.float32)
rot = euler_to_so3(leftlidar2cam[3:6])
trs = np.array(leftlidar2cam[:3])
leftmatrixR[:3, :3] = rot
leftmatrixR[0, 3] = trs[0]
leftmatrixR[1, 3] = trs[1]
leftmatrixR[2, 3] = trs[2]
leftmatrixR[3, 3] = 1

rightmatrixR = np.zeros((4,4), dtype=np.float32)
rot = euler_to_so3(rightlidar2cam[3:6])
trs = np.array(rightlidar2cam[:3])
rightmatrixR[:3, :3] = rot
rightmatrixR[0, 3] = trs[0]
rightmatrixR[1, 3] = trs[1]
rightmatrixR[2, 3] = trs[2]
rightmatrixR[3, 3] = 1


def check_vaild_and_detect_road_points_left(pointcloud):
    pointcloud = pointcloud[:, :3]
    mask = (pointcloud[:, 0] >= cfg.x_roi[0]) & (pointcloud[:, 0] <= cfg.x_roi[1])\
         & (pointcloud[:, 1] >= cfg.y_roi[0]) & (pointcloud[:, 1] <= cfg.y_roi[1])

    pointcloud = pointcloud[mask]
    mask = (pointcloud[:, 2] >= cfg.z_roi[0]) & (pointcloud[:, 2] <= cfg.z_roi[1])
    pointcloud = pointcloud[mask]

    mask = (pointcloud[:, 0]  < 5) & (pointcloud[:, 0] > -5)\
        & (pointcloud[:, 1]  < 5) & (pointcloud[:, 1] > -5)
    mask = np.logical_not(mask)
    pointcloud = pointcloud[mask]

    return pointcloud.copy()


def check_vaild_and_detect_road_points_right(pointcloud):
    pointcloud = pointcloud[:, :3]
    mask = (pointcloud[:, 0] >= cfg.x_roi[0]) & (pointcloud[:, 0] <= cfg.x_roi[1])\
         & (pointcloud[:, 1] >= cfg.y_roi[0]) & (pointcloud[:, 1] <= cfg.y_roi[1])

    pointcloud = pointcloud[mask]
    mask = (pointcloud[:, 2] >= cfg.z_roi[0]) & (pointcloud[:, 2] <= cfg.z_roi[1])
    pointcloud = pointcloud[mask]

    mask = (pointcloud[:, 0]  < 1.5) & (pointcloud[:, 0] > -2.5)\
        & (pointcloud[:, 1]  < 1.5) & (pointcloud[:, 1] > -1.5)
    mask = np.logical_not(mask)
    pointcloud = pointcloud[mask]

    return pointcloud.copy()



scene1_left = cfg.oxford_home + '2019-01-10-12-32-52-radar-oxford-10k_Velodyne_HDL-32E_Left_Pointcloud-001/2019-01-10-12-32-52-radar-oxford-10k/velodyne_left/'
scene1_right = cfg.oxford_home + '2019-01-10-12-32-52-radar-oxford-10k_Velodyne_HDL-32E_Right_Pointcloud-001/2019-01-10-12-32-52-radar-oxford-10k/velodyne_right/'
scene1_vo = cfg.oxford_home + '2019-01-10-12-32-52-radar-oxford-10k_Bumblebee_XB3_Visual_Odometry/2019-01-10-12-32-52-radar-oxford-10k/vo/vo.csv'
scene2_left = cfg.oxford_home + '2019-01-17-12-48-25-radar-oxford-10k_Velodyne_HDL-32E_Left_Pointcloud-001/2019-01-17-12-48-25-radar-oxford-10k/velodyne_left/'
scene2_right = cfg.oxford_home + '2019-01-17-12-48-25-radar-oxford-10k_Velodyne_HDL-32E_Right_Pointcloud-002/2019-01-17-12-48-25-radar-oxford-10k/velodyne_right/'
scene2_vo = cfg.oxford_home + '2019-01-17-12-48-25-radar-oxford-10k_Bumblebee_XB3_Visual_Odometry/2019-01-17-12-48-25-radar-oxford-10k/vo/vo.csv'
outhome = cfg.proj_home + cfg.data_subdir

if os.path.exists(outhome + 'scene1/'):
    shutil.rmtree(outhome + 'scene1/')
os.makedirs(outhome + 'scene1/left/')
os.makedirs(outhome + 'scene1/right/')

if os.path.exists(outhome + 'scene2/'):
    shutil.rmtree(outhome + 'scene2/')
os.makedirs(outhome + 'scene2/left/')
os.makedirs(outhome + 'scene2/right/')


scene1_left_list = os.listdir(scene1_left)
scene1_right_list = os.listdir(scene1_right)
scene2_left_list = os.listdir(scene2_left)
scene2_right_list = os.listdir(scene2_right)

scene1_left_list.sort()
scene1_right_list.sort()
scene2_left_list.sort()
scene2_right_list.sort()


with open(cfg.proj_home + 'gendata/' + cfg.traintxt, mode='r') as f:
   lidar_scene1_times = f.readlines()

odometry_scene1_vocsv = pd.read_csv(scene1_vo)
odometry_scene1_vocsv = odometry_scene1_vocsv[1:].to_numpy()
odometry_scene1_times = odometry_scene1_vocsv[:, :2].astype(np.uint64)
odometry_scene1_xyz = odometry_scene1_vocsv[:, 2:5].astype(np.float64)
odometry_scene1_rpy = odometry_scene1_vocsv[:, 5:].astype(np.float64)

scene1_init_cnt = 0
scene1_lines = lidar_scene1_times[0].replace("\n", "")
time2, time1 = scene1_lines.split(' ')
time2, time1 = int(time2), int(time1)
for i in range(odometry_scene1_times.shape[0]):
   if time1 - odometry_scene1_times[i, 1] < 0:
       scene1_init_cnt = scene1_init_cnt - 1
       break
   scene1_init_cnt = scene1_init_cnt + 1
   continue

with open(cfg.proj_home + 'gendata/oxford-train.txt', mode='w') as f:
    for lidar_idx in range(0, len(lidar_scene1_times)):
        print(lidar_idx)
        lines = lidar_scene1_times[lidar_idx].replace("\n", "")

        time1, time2 = lines.split(' ')
        leftbin = np.fromfile(scene1_left + time1 + '.bin', dtype=np.float32).reshape(4, -1)
        rightbin = np.fromfile(scene1_right + time2 + '.bin', dtype=np.float32).reshape(4, -1)
        leftbin[3, :] = 1
        rightbin[3, :] = 1

        time1, time2 = int(time1), int(time2)

        init_dif = time1 - odometry_scene1_times[scene1_init_cnt, 1]
        cnt_move = False
        while True:
            next_dif = time1 - odometry_scene1_times[scene1_init_cnt + 1, 1]
            if next_dif < 0:
                break

            if next_dif < init_dif:
                init_dif = next_dif
                scene1_init_cnt = scene1_init_cnt + 1
            continue

        odometry_btw_time = odometry_scene1_times[scene1_init_cnt, 0] - odometry_scene1_times[scene1_init_cnt, 1]
        lidar_btw_time = time1 - time2
        scale = lidar_btw_time / odometry_btw_time
        newx = odometry_scene1_xyz[scene1_init_cnt, 0] * scale
        newy = odometry_scene1_xyz[scene1_init_cnt, 1] * scale
        newz = odometry_scene1_xyz[scene1_init_cnt, 2] * scale
        newroll = odometry_scene1_rpy[scene1_init_cnt, 0] * scale
        newpitch = odometry_scene1_rpy[scene1_init_cnt, 1] * scale
        newyaw = odometry_scene1_rpy[scene1_init_cnt, 2] * scale

        move_ = [newx, newy, newz, newroll, newpitch, newyaw]
        moveR = np.zeros((4,4), dtype=np.float32)
        rot = euler_to_so3(move_[3:6])
        trs = np.array(move_[:3])
        moveR[:3, :3] = rot
        moveR[0, 3] = trs[0]
        moveR[1, 3] = trs[1]
        moveR[2, 3] = trs[2]
        moveR[3, 3] = 1

        leftbin = np.dot(leftmatrixR, leftbin)
        leftbin = np.dot(moveR, leftbin)
        rightbin = np.dot(rightmatrixR, rightbin)

        leftbin = check_vaild_and_detect_road_points_left(leftbin.T)
        rightbin = check_vaild_and_detect_road_points_right(rightbin.T)

        leftbin.tofile(outhome + 'scene1/left/' + str(time1) + '.bin')
        rightbin.tofile(outhome + 'scene1/right/' + str(time2) + '.bin')
        f.write("%d %d\n" % (time1, time2))


        # if newx > 0.5 or newy > 0.1 or newz > 0.1:
        # points = np.concatenate((leftbin, rightbin), axis=0)

        # leftcolormap = np.zeros((leftbin.shape[0], 3), dtype=np.float32)
        # rightcolormap = np.zeros((rightbin.shape[0], 3), dtype=np.float32)

        # leftcolormap[:, 0] = leftcolormap[:, 0] + 0.999
        # rightcolormap[:, 1] = rightcolormap[:, 1] + 0.999
        
        # colormap = np.concatenate((leftcolormap, rightcolormap), axis=0)

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colormap)
        # o3d.visualization.draw_geometries([pcd])
        # df=df

with open(cfg.proj_home + 'gendata/' + cfg.valtxt, mode='r') as f:
   lidar_scene2_times = f.readlines()


odometry_scene2_vocsv = pd.read_csv(scene2_vo)
odometry_scene2_vocsv = odometry_scene2_vocsv[1:].to_numpy()
odometry_scene2_times = odometry_scene2_vocsv[:, :2].astype(np.uint64)
odometry_scene2_xyz = odometry_scene2_vocsv[:, 2:5].astype(np.float64)
odometry_scene2_rpy = odometry_scene2_vocsv[:, 5:].astype(np.float64)

scene2_init_cnt = 0
scene2_lines = lidar_scene2_times[0].replace("\n", "")
time2, time1 = scene2_lines.split(' ')
time2, time1 = int(time2), int(time1)
for i in range(odometry_scene2_times.shape[0]):
   if time1 - odometry_scene2_times[i, 1] < 0:
       scene2_init_cnt = scene2_init_cnt - 1
       break
   scene2_init_cnt = scene2_init_cnt + 1
   continue

with open(cfg.proj_home + 'gendata/oxford-val.txt', mode='w') as f:
    for lidar_idx in range(len(lidar_scene2_times)):
        if lidar_idx == odometry_scene2_times.shape[0]:
            break

        print(lidar_idx)
        lines = lidar_scene2_times[lidar_idx].replace("\n", "")

        time1, time2 = lines.split(' ')
        leftbin = np.fromfile(scene2_left + time1 + '.bin', dtype=np.float32).reshape(4, -1)
        rightbin = np.fromfile(scene2_right + time2 + '.bin', dtype=np.float32).reshape(4, -1)
        leftbin[3, :] = 1
        rightbin[3, :] = 1

        time1, time2 = int(time1), int(time2)

        init_dif = time1 - odometry_scene2_times[scene2_init_cnt, 1]
        cnt_move = False
        while True:
            next_dif = time1 - odometry_scene2_times[scene2_init_cnt + 1, 1]
            if next_dif < 0:
                break

            if next_dif < init_dif:
                init_dif = next_dif
                scene2_init_cnt = scene2_init_cnt + 1
            continue

        odometry_btw_time = odometry_scene2_times[scene2_init_cnt, 0] - odometry_scene2_times[scene2_init_cnt, 1]
        lidar_btw_time = time1 - time2
        scale = lidar_btw_time / odometry_btw_time
        newx = odometry_scene2_xyz[scene2_init_cnt, 0] * scale
        newy = odometry_scene2_xyz[scene2_init_cnt, 1] * scale
        newz = odometry_scene2_xyz[scene2_init_cnt, 2] * scale
        newroll = odometry_scene2_rpy[scene2_init_cnt, 0] * scale
        newpitch = odometry_scene2_rpy[scene2_init_cnt, 1] * scale
        newyaw = odometry_scene2_rpy[scene2_init_cnt, 2] * scale

        move_ = [newx, newy, newz, newroll, newpitch, newyaw]
        moveR = np.zeros((4,4), dtype=np.float32)
        rot = euler_to_so3(move_[3:6])
        trs = np.array(move_[:3])
        moveR[:3, :3] = rot
        moveR[0, 3] = trs[0]
        moveR[1, 3] = trs[1]
        moveR[2, 3] = trs[2]
        moveR[3, 3] = 1

        leftbin = np.dot(leftmatrixR, leftbin)
        leftbin = np.dot(moveR, leftbin)
        rightbin = np.dot(rightmatrixR, rightbin)

        leftbin = check_vaild_and_detect_road_points_left(leftbin.T)
        rightbin = check_vaild_and_detect_road_points_right(rightbin.T)

        leftbin.tofile(outhome + 'scene2/left/' + str(time1) + '.bin')
        rightbin.tofile(outhome + 'scene2/right/' + str(time2) + '.bin')
        f.write("%d %d\n" % (time1, time2))

        # if newx > 0.2 or newy > 0.1 or newz > 0.1:
        #     print(newx, newy, newz)
        #     points = np.concatenate((leftbin, rightbin), axis=0)

        #     leftcolormap = np.zeros((leftbin.shape[0], 3), dtype=np.float32)
        #     rightcolormap = np.zeros((rightbin.shape[0], 3), dtype=np.float32)

        #     leftcolormap[:, 0] = leftcolormap[:, 0] + 0.999
        #     rightcolormap[:, 1] = rightcolormap[:, 1] + 0.999
            
        #     colormap = np.concatenate((leftcolormap, rightcolormap), axis=0)

        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(points)
        #     pcd.colors = o3d.utility.Vector3dVector(colormap)
        #     o3d.visualization.draw_geometries([pcd])
