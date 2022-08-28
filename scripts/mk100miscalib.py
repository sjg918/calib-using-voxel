
import numpy as np

error_list = []
for i in range(100):
    rot_x = np.random.uniform(-20, 20)
    rot_y = np.random.uniform(-20, 20)
    rot_z = np.random.uniform(-20, 20)
    trs_x = np.random.uniform(-1.5, 1.5)
    trs_y = np.random.uniform(-1.5, 1.5)
    trs_z = np.random.uniform(-1.5, 1.5)
    error_list.append([rot_x, rot_y, rot_z, trs_x, trs_y, trs_z])
    
with open('/home/jnu-ie/kew/calib-using-voxel/gendata/100miscalib.txt', 'w') as writer:
    for i in range(100):
        rot_x, rot_y, rot_z, trs_x, trs_y, trs_z = error_list[i]
        if i==99:
            writer.write(str(rot_x) + ' ' + str(rot_y) + ' ' + str(rot_z) + ' ' + str(trs_x) + ' ' + str(trs_y) + ' ' + str(trs_z))
            continue
        writer.write(str(rot_x) + ' ' + str(rot_y) + ' ' + str(rot_z) + ' ' + str(trs_x) + ' ' + str(trs_y) + ' ' + str(trs_z) + '\n')
        continue
