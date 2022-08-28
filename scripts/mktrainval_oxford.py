
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.CFG_oxford20 import cfg

scene1_left = cfg.oxford_home + '2019-01-10-12-32-52-radar-oxford-10k_Velodyne_HDL-32E_Left_Pointcloud-001/2019-01-10-12-32-52-radar-oxford-10k/velodyne_left/'
scene1_right = cfg.oxford_home + '2019-01-10-12-32-52-radar-oxford-10k_Velodyne_HDL-32E_Right_Pointcloud-001/2019-01-10-12-32-52-radar-oxford-10k/velodyne_right/'
scene2_left = cfg.oxford_home + '2019-01-17-12-48-25-radar-oxford-10k_Velodyne_HDL-32E_Left_Pointcloud-001/2019-01-17-12-48-25-radar-oxford-10k/velodyne_left/'
scene2_right = cfg.oxford_home + '2019-01-17-12-48-25-radar-oxford-10k_Velodyne_HDL-32E_Right_Pointcloud-002/2019-01-17-12-48-25-radar-oxford-10k/velodyne_right/'

scene1_left_list = os.listdir(scene1_left)
scene1_right_list = os.listdir(scene1_right)
scene2_left_list = os.listdir(scene2_left)
scene2_right_list = os.listdir(scene2_right)

scene1_left_list.sort()
scene1_right_list.sort()
scene2_left_list.sort()
scene2_right_list.sort()

# print(len(scene1_left_list))
# print(datetime.fromtimestamp(int(scene1_left_list[12].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene1_right_list[12-12].split('.')[0]) / 1000000))
# print(datetime.fromtimestamp(int(scene1_left_list[13].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene1_right_list[1].split('.')[0]) / 1000000))
# print(datetime.fromtimestamp(int(scene1_left_list[14].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene1_right_list[2].split('.')[0]) / 1000000))
# print(datetime.fromtimestamp(int(scene1_left_list[15].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene1_right_list[3].split('.')[0]) / 1000000))
# print(datetime.fromtimestamp(int(scene1_left_list[16].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene1_right_list[4].split('.')[0]) / 1000000))

# print(len(scene2_left_list))
# print(datetime.fromtimestamp(int(scene2_left_list[12].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene2_right_list[12-12].split('.')[0]) / 1000000))
# print(datetime.fromtimestamp(int(scene2_left_list[13].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene2_right_list[1].split('.')[0]) / 1000000))
# print(datetime.fromtimestamp(int(scene2_left_list[14].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene2_right_list[2].split('.')[0]) / 1000000))
# print(datetime.fromtimestamp(int(scene2_left_list[15].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene2_right_list[3].split('.')[0]) / 1000000))
# print(datetime.fromtimestamp(int(scene2_left_list[16].split('.')[0]) / 1000000), datetime.fromtimestamp(int(scene2_right_list[4].split('.')[0]) / 1000000))

with open(cfg.proj_home + 'gendata/' + cfg.traintxt, mode='w') as f:
    for cnt, seq in enumerate(scene1_left_list):
        if cnt < 13:
            continue

        leftnum = seq.split('.')[0]
        rightnum = scene1_right_list[cnt - 12].split('.')[0]

        f.write(leftnum + ' ' + rightnum +  '\n')


with open(cfg.proj_home + 'gendata/' + cfg.valtxt, mode='w') as f:
    for cnt, seq in enumerate(scene2_left_list):
        if cnt < 13:
            continue

        leftnum = seq.split('.')[0]
        rightnum = scene2_right_list[cnt - 12].split('.')[0]

        f.write(leftnum + ' ' + rightnum +  '\n')
