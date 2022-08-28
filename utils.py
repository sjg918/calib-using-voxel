
import numpy as np
import math
import torch
import numba
import matplotlib.pyplot as plt

#refer : https://learnopencv.com/rotation-matrix-to-euler-angles/
# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])


#refer : https://github.com/IIPCVLAB/LCCNet
def quaternion_from_rotation_matrix(matrix):
    if matrix.shape == (4, 4):
        R = matrix[:3, :3]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = np.zeros(4, dtype=np.float32)
    if tr > 0.:
        S = np.sqrt(tr+1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / np.linalg.norm(q)


def rotation_matrix_from_quaternion(q):
    mat = np.zeros((3,3), dtype=np.float32)

    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    return mat


# refer : https://github.com/utiasSTARS/pykitti
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def rot_points(points, R):
    newpoints = []
    for p in points:
        kew = np.dot(R, p.T)
        newpoints.append(kew)
    return np.array(newpoints)


def trs_points(points, trs):
    newpoints = []
    for p in points:
        kew = p + trs
        newpoints.append(kew)
    return np.array(newpoints)


@numba.jit
def rot_and_trs_points_kernel(points, newpoints, R, numpoints):
    for i in range(numpoints):
        p = points[i]
        kew = np.dot(R, p.T)
        newpoints[i, 0] = kew[0]
        newpoints[i, 1] = kew[1]
        newpoints[i, 2] = kew[2]
        continue

def rot_and_trs_points(points ,R):
    o = np.ones((points.shape[0], 1), dtype=np.float32)
    points = np.concatenate((points, o), axis=1)
    newpoints = np.zeros((points.shape[0], 3), dtype=np.float32)
    rot_and_trs_points_kernel(points.astype(np.float32), newpoints, R.astype(np.float32), points.shape[0])
    return newpoints


def steploss_string(cnt, datalen, all, rot, trs,):
    return "{}/{} losses:total{:.5f} rot{:.5f} trs{:.5f}".format(
                cnt, datalen,
                sum(all) / len(all),
                sum(rot) / len(rot),
                sum(trs) / len(trs),
            )


def load_model(m, p, device):
    dict = torch.load(p, map_location=device)
    for i, k in zip(m.state_dict(), dict):
        weight = dict[k]
        m.state_dict()[i].copy_(weight)
        
        
def eval_100miscalib(path, misclaibtxtpath, kitti=True):
    error_rotx = []
    error_roty = []
    error_rotz = []
    error_trsx = []
    error_trsy = []
    error_trsz = []

    with open(misclaibtxtpath, 'r') as f:
        predefine_error_list = f.readlines()

    for number in range(100):
        pred_median = np.load(path + 'pred_median_' + str(number) + '.npy')

        velo_rotx, velo_roty, velo_rotz, velo_trsx, velo_trsy, velo_trsz = predefine_error_list[number].strip().split(' ')
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

        predrotmat = eulerAnglesToRotationMatrix([pred_median[0] * (3.141592 / 180.0), pred_median[1] * (3.141592 / 180.0), pred_median[2] * (3.141592 / 180.0)])
        calib_RT_running = np.zeros((4, 4), dtype=np.float32)
        calib_RT_running[:3, :3] = predrotmat[:3,:3]
        calib_RT_running[0, 3] = pred_median[3]
        calib_RT_running[1, 3] = pred_median[4]
        calib_RT_running[2, 3] = pred_median[5]
        calib_RT_running[3, 3] = 1

        # cal error
        error_MAT = calib_RT_running @ misRTmat
        error_ROT = rotationMatrixToEulerAngles(error_MAT[:3, :3]) * (180.0 / np.pi)
        error_TRS = error_MAT[:3, 3]

        error_rot = np.abs(error_ROT)
        error_trs = np.abs(error_TRS)
        
        if kitti:
            error_rotx.append(error_rot[2])
            error_roty.append(error_rot[0])
            error_rotz.append(error_rot[1])
        else:
            error_rotx.append(error_rot[0])
            error_roty.append(error_rot[1])
            error_rotz.append(error_rot[2])
        error_trsx.append(error_trs[0])
        error_trsy.append(error_trs[1])
        error_trsz.append(error_trs[2])

        #print(number, error_rot, error_trs)
    
    error_rotx = np.array(error_rotx)
    error_roty = np.array(error_roty)
    error_rotz = np.array(error_rotz)
    error_trsx = np.array(error_trsx)
    error_trsy = np.array(error_trsy)
    error_trsz = np.array(error_trsz)
    
    print(np.mean(error_rotx), np.mean(error_roty), np.mean(error_rotz), np.mean(error_trsx), np.mean(error_trsy), np.mean(error_trsz))

    fig, ax1 = plt.subplots(figsize=(4,4), dpi=160)
    ax1.set_xlabel('Rotation Error in °')
    ax1.boxplot([error_rotx, error_roty, error_rotz], showfliers=False)

    plt.ylim([-0.5, 0.5])
    plt.xticks([1,2,3],['Roll', 'Pitch', 'Yaw'])
    plt.savefig(path + 'kitti-rotation-boxplot-.png', bbox_inches='tight', pad_inches=0)

    fig, ax2 = plt.subplots(figsize=(4,4), dpi=160)
    ax2.set_xlabel('Translation Error in m')
    ax2.boxplot([error_trsx, error_trsy, error_trsz], showfliers=False)

    plt.ylim([-0.05, 0.05])
    plt.xticks([1,2,3],['X', 'Y', 'Z'])
    plt.savefig(path + '/kitti-trnaslation-boxplot_.png', bbox_inches='tight', pad_inches=0)
