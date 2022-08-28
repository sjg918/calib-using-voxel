
import numpy as np
import cv2
import os
import shutil
from pykitti import odometry
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.CFG_kitti20 import cfg
from src.sgmgpu.utils import mkDispmap
import torch

home = cfg.odometry_home + 'data_odometry_color/dataset/sequences/'
velohome = cfg.odometry_home + 'data_odometry_velodyne/dataset/sequences/'
#sgmhome = cfg.proj_home + 'gendata/data_odometry_sgm/dataset/sequences/'
outhome = cfg.proj_home + cfg.data_subdir

sequencelist = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']


for seq in sequencelist:
    print(seq)
    if os.path.exists(outhome + str(seq) + '/disp_' + str(cfg.sgm_path)  + '/'):
        shutil.rmtree(outhome + str(seq) + '/disp_' + str(cfg.sgm_path)  + '/')
    os.makedirs(outhome + str(seq) + '/disp_' + str(cfg.sgm_path)  + '/')

    if os.path.exists(outhome + str(seq) + '/velodyne_' + str(cfg.lidar_channel)  + '/'):
        shutil.rmtree(outhome + str(seq) + '/velodyne_' + str(cfg.lidar_channel)  + '/')
    os.makedirs(outhome + str(seq) + '/velodyne_' + str(cfg.lidar_channel)  + '/')


def check_vaild_and_detect_road_points(pointcloud):
    pointcloud = pointcloud[:, :3]
    mask = (pointcloud[:, 0] >= cfg.lidarx_roi[0]) & (pointcloud[:, 0] <= cfg.lidarx_roi[1])\
         & (pointcloud[:, 1] >= cfg.lidary_roi[0]) & (pointcloud[:, 1] <= cfg.lidary_roi[1])

    pointcloud = pointcloud[mask]
    mask = (pointcloud[:, 2] >= cfg.lidarz_roi[0]) & (pointcloud[:, 2] <= cfg.lidarz_roi[1])
    pointcloud = pointcloud[mask]
    return pointcloud.copy()


def project_disp_to_points_and_detect_road(sgmdispmap, K):
    c_u = K[0, 2]
    c_v = K[1, 2]
    f_u = K[0, 0]
    f_v = K[1, 1]
    baseline = 0.54

    sgmdispmap[sgmdispmap < 0] = 0
    mask = sgmdispmap > 0
    depth = f_u * baseline / (sgmdispmap + 1. - mask)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    points = points[mask.reshape(-1)]

    n = points.shape[0]
    x = ((points[:, 0] - c_u) * points[:, 2]) / f_u
    y = ((points[:, 1] - c_v) * points[:, 2]) / f_v
    pts_3d_rect = np.zeros((n, 3), dtype=np.float32)
    pts_3d_rect[:, 0] = x
    pts_3d_rect[:, 1] = y
    pts_3d_rect[:, 2] = points[:, 2]

    pts_3d_rect = get_sparse_pseudo_lidar(pts_3d_rect)

    mask = (pts_3d_rect[:, 2] >= cfg.camz_roi[0]) & (pts_3d_rect[:, 2] <= cfg.camz_roi[1])\
         & (pts_3d_rect[:, 0] >= cfg.camx_roi[0]) & (pts_3d_rect[:, 0] <= cfg.camx_roi[1])
    pts_3d_rect = pts_3d_rect[mask]
    mask = (pts_3d_rect[:, 1] >= cfg.camy_roi[0]) & (pts_3d_rect[:, 1] <= cfg.camy_roi[1])
    pts_3d_rect = pts_3d_rect[mask]
    return pts_3d_rect.copy()

  
# refer : https://github.com/chenfengxu714/SqueezeSegV3
def get_low_channel_lidar(scan, H=64, W=2048, fov_up=3.0, fov_down=-25.0):
    if cfg.lidar_channel == 64:
        return scan

    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission

    # projected range image - [H,W] range (-1 is no data)
    proj_range = np.full((H, W), -1, dtype=np.float32)

    # unprojected range (list of depths for each point)
    unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_xyz = np.full((H, W, 3), -1, dtype=np.float32)

    # projected remission - [H,W] intensity (-1 is no data)
    proj_remission = np.full((H, W), -1, dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((H, W), -1, dtype=np.int32)

    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    proj_mask = np.zeros((H, W), dtype=np.int32)       # [H,W] mask

    """ Project a pointcloud into a spherical projection image.projection.
    Function takes no arguments because it can be also called externally
    if the value of the constructor was not set (in case you change your
    mind about wanting the projection)
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= W                              # in [0.0, W]
    proj_y *= H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    #proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    #proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    #unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    remission = remissions[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = points
    proj_remission[proj_y, proj_x] = remission
    proj_idx[proj_y, proj_x] = indices
    #proj_mask = (proj_idx > 0).astype(np.int32)
    proj_mask = proj_idx > 0

    if cfg.lidar_channel == 8:
        for i in range(64):
            if i % 8 != 0:
                proj_idx[i, :] = -1
        proj_mask = proj_idx > 0
        vaildmask = proj_idx[proj_mask]
        new_scan = scan[vaildmask, :]
        return new_scan.copy()
    elif cfg.lidar_channel == 16:
        for i in range(64):
            if i % 4 != 0:
                proj_idx[i, :] = -1
        proj_mask = proj_idx > 0
        vaildmask = proj_idx[proj_mask]
        new_scan = scan[vaildmask, :]
        return new_scan.copy()
    elif cfg.lidar_channel == 32:
        for i in range(64):
            if i % 2 != 0:
                proj_idx[i, :] = -1
        proj_mask = proj_idx > 0
        vaildmask = proj_idx[proj_mask]
        new_scan = scan[vaildmask, :]
        return new_scan.copy()


def get_sparse_pseudo_lidar(scan, H=128, W=2048, fov_up=3.0, fov_down=-25.0):
    points = scan[:, 0:3]    # get xyz

    # projected range image - [H,W] range (-1 is no data)
    proj_range = np.full((H, W), -1, dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    proj_xyz = np.full((H, W, 3), -1, dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    proj_idx = np.full((H, W), -1, dtype=np.int32)

    # for each point, where it is in the range image
    proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    proj_mask = np.zeros((H, W), dtype=np.int32)       # [H,W] mask

    """ Project a pointcloud into a spherical projection image.projection.
    Function takes no arguments because it can be also called externally
    if the value of the constructor was not set (in case you change your
    mind about wanting the projection)
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 2]
    scan_y = points[:, 0] * -1
    scan_z = points[:, 1] * -1

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= W                              # in [0.0, W]
    proj_y *= H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    #proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    #proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    #unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = points[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = points
    proj_idx[proj_y, proj_x] = indices
    #proj_mask = (proj_idx > 0).astype(np.int32)
    proj_mask = proj_idx > 0

    proj_mask = proj_idx > 0
    vaildmask = proj_idx[proj_mask]
    new_scan = scan[vaildmask, :]
    return new_scan.copy()


sgm_timelist = []
convert_timelist = []
list_num_road_points_from_sgm = []
list_num_road_points_from_velo = []
for seq in sequencelist:
    odom = odometry(cfg.odometry_home + 'data_odometry_color/dataset/', seq)
    calib = odom.calib
    K = calib.K_cam2

    velo_path = velohome + str(seq) + '/velodyne/'
    left_path = home + str(seq) + '/image_2/'
    right_path = home + str(seq) + '/image_3/'
    
    sgmout_path = outhome + str(seq) + '/disp_' + str(cfg.sgm_path)  + '/'
    veloout_path = outhome + str(seq) + '/velodyne_' + str(cfg.lidar_channel)  + '/'

    imgnum_list = os.listdir(left_path)
    for imgnum in imgnum_list:
        imgnum = imgnum[:-4]

        left_img = cv2.imread(left_path + imgnum + '.png')
        right_img = cv2.imread(right_path + imgnum + '.png')
        h, w, _ = left_img.shape

        top_pad = 384 - h
        right_pad = 1248 - w
        assert top_pad > 0 and right_pad > 0
        left_img = np.lib.pad(left_img, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((top_pad, 0), (0, right_pad), (0, 0)), mode='constant',constant_values=0)

        left_img_ = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_img_ = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        left_img = torch.from_numpy(left_img_)
        right_img = torch.from_numpy(right_img_)
        left_img = left_img.to(cfg.device)
        right_img = right_img.to(cfg.device)

        start_time = time.time()
        disp_img = mkDispmap(left_img, right_img, cfg.sgm_path, return_gpu=False)
        sgm_timelist.append(time.time() - start_time)

        disp_img = disp_img.numpy()
        if top_pad !=0 or right_pad != 0:
            disp_img = disp_img[top_pad:,:-right_pad]

        start_time = time.time()
        road_points_from_sgm = project_disp_to_points_and_detect_road(disp_img, K)
        convert_timelist.append(time.time() - start_time)

        pointcloud = np.fromfile(velo_path + imgnum + '.bin', dtype=np.float32).reshape((-1, 4))
        pointcloud = get_low_channel_lidar(pointcloud)
        road_points_from_velo = check_vaild_and_detect_road_points(pointcloud)


        list_num_road_points_from_sgm.append(road_points_from_sgm.shape[0])
        list_num_road_points_from_velo.append(road_points_from_velo.shape[0])
        road_points_from_sgm.tofile(sgmout_path + imgnum + '.bin')
        road_points_from_velo.tofile(veloout_path + imgnum + '.bin')

        continue
    continue
print(sum(sgm_timelist) / len(sgm_timelist))
print(sum(convert_timelist) / len(convert_timelist))
print(sum(list_num_road_points_from_sgm) / len(list_num_road_points_from_sgm))
#74216.89580271859
print(sum(list_num_road_points_from_velo) / len(list_num_road_points_from_velo))
#18145.22306667891
