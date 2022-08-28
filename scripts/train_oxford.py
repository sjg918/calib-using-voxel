
import random
import logging
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from spconv.pytorch.utils import PointToVoxel
import spconv.pytorch as spconv

import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from CFG.CFG_oxford20 import cfg
from src.voxelnet import Deepv2_base
from src.oxfordfactory import DataFactory
from src.utils import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visible

def train():
    # start
    print(cfg.model)
    # define model
    model = Deepv2_base(cfg.left_local, cfg.right_local, split=False, dropout=True).to(cfg.device)
    #model = Deepv2_newbase(cfg.left_local, cfg.right_local, split=False, dropout=True, lambda1=cfg.lambda1, lambda2=cfg.lambda2).to(cfg.device)
    
    point2voxel = PointToVoxel(vsize_xyz=cfg.voxelsize,
                           coors_range_xyz=cfg.voxelrange,
                           num_point_features=4,
                           max_num_voxels=cfg.maxvoxels,
                           max_num_points_per_voxel=cfg.maxpoints,
                           device=cfg.device)

    # define dataloader
    kitti_dataset = DataFactory(cfg)
    kitti_loader = DataLoader(kitti_dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=cfg.num_cpu,\
       pin_memory=True, drop_last=True, collate_fn=kitti_dataset.collate_fn_cpu)

    # define optimizer and scheduler
    model_optimizer = optim.Adam(model.parameters(), lr=cfg.learing_rate, betas=(0.9, 0.999), eps=1e-08)
    model_scheduler = optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=cfg.MultiStepLR_milstone, gamma=cfg.MultiStepLR_gamma)

    # define loss function
    # mmodel contain losses!

    model.train()
    runningloss_all = []
    runningloss_rot = []
    runningloss_trs = []
    
    for epoch in range(1, cfg.maxepoch+1):
        # print milestones
        print('({} / {}) epoch\n'.format(epoch, cfg.maxepoch))
        with open(cfg.logdir + 'log.txt', 'a') as writer:
            writer.write('({} / {}) epoch\n'.format(epoch, cfg.maxepoch))

        for cnt, traindatadict in enumerate(kitti_loader):
            # pre process

            sgm_list = traindatadict['right']
            velo_list = traindatadict['left']

            sgm_list = [i.to(dtype=torch.float32, device=cfg.device) for i in sgm_list]
            velo_list = [i.to(dtype=torch.float32, device=cfg.device) for i in velo_list]
            sgm_voxels_list = []
            sgm_coors_list = []
            velo_voxels_list = []
            velo_coors_list = []

            b = 0
            for sgm, velo in zip(sgm_list, velo_list):
                sgm_voxels, sgm_indices, sgm_num_p_in_vx = point2voxel(sgm)
                sgm_voxels = sgm_voxels[:, :, :3].sum(dim=1, keepdim=False) / sgm_num_p_in_vx.type_as(sgm_voxels).view(-1, 1)
                sgm_batch = torch.zeros((sgm_num_p_in_vx.shape[0], 1), dtype=torch.int32, device=cfg.device) + b
                sgm_coors = sgm_indices.to(dtype=torch.int32)
                sgm_coors = torch.cat((sgm_batch, sgm_coors), dim=1)

                velo_voxels, velo_indices, velo_num_p_in_vx = point2voxel(velo)
                velo_voxels = velo_voxels[:, :, :3].sum(dim=1, keepdim=False) / velo_num_p_in_vx.type_as(velo_voxels).view(-1, 1)
                velo_batch = torch.zeros((velo_num_p_in_vx.shape[0], 1), dtype=torch.int32, device=cfg.device) + b
                velo_coors = velo_indices.to(dtype=torch.int32)
                velo_coors = torch.cat((velo_batch, velo_coors), dim=1)

                b = b + 1
                sgm_voxels_list.append(sgm_voxels)
                sgm_coors_list.append(sgm_coors)
                velo_voxels_list.append(velo_voxels)
                velo_coors_list.append(velo_coors)

            sgm_voxels_list = torch.cat(sgm_voxels_list, dim=0)
            sgm_coors_list = torch.cat(sgm_coors_list, dim=0)
            sgm_input = spconv.SparseConvTensor(sgm_voxels_list, sgm_coors_list, cfg.voxelshape, cfg.batchsize)

            velo_voxels_list = torch.cat(velo_voxels_list, dim=0)
            velo_coors_list = torch.cat(velo_coors_list, dim=0)
            velo_input = spconv.SparseConvTensor(velo_voxels_list, velo_coors_list, cfg.voxelshape, cfg.batchsize)

            targetQuaternion = traindatadict['tgtQ']
            targetTranslate = traindatadict['tgtT']
            target = {
                'quaternion': targetQuaternion.to(dtype=torch.float32, device=cfg.device),
                'translate': targetTranslate.to(dtype=torch.float32, device=cfg.device)
            }
                            
            # forward
            loss, rotloss, trsloss = model(sgm_input, velo_input, target)
    
            # backward
            loss.backward()
            model_optimizer.step()
            model.zero_grad()
            
            runningloss_all.append(loss.item())
            runningloss_rot.append(rotloss.item())
            runningloss_trs.append(trsloss.item())

            # print steploss
            print(steploss_string(cnt, len(kitti_dataset) / cfg.batchsize, runningloss_all, runningloss_rot, runningloss_trs), end="\r")
            continue

        # learning rate scheduling
        model_scheduler.step()
        
        print(steploss_string(cnt, len(kitti_dataset) / cfg.batchsize, runningloss_all, runningloss_rot, runningloss_trs) + '\n')
        with open(cfg.logdir + 'log.txt', 'a') as writer:
            writer.write(steploss_string(cnt, len(kitti_dataset) / cfg.batchsize, runningloss_all, runningloss_rot, runningloss_trs) + '\n')

        runningloss_all = []
        runningloss_rot = []
        runningloss_trs = []

        # save model
        if epoch % 5 == 0:
            torch.save(model.state_dict(), cfg.logdir + 'model_' + str(epoch) + '.pth')
            #torch.save(back_optimizer.state_dict(), cfg.logdir + '/backopti_' + str(epoch) + '.pth')
            #torch.save(neck_optimizer.state_dict(), cfg.logdir + '/neckopti_' + str(epoch) + '.pth')
            print('{} epoch model saved !\n'.format(epoch))
            with open(cfg.logdir + 'log.txt', 'a') as writer:
                writer.write('{} epoch model saved !\n'.format(epoch))
        continue
    # end.

if __name__ == '__main__':
    torch.manual_seed(7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(7)
    np.random.seed(7)

    if os.path.exists(cfg.logdir):
        shutil.rmtree(cfg.logdir)
    os.makedirs(cfg.logdir)

    with open(cfg.logdir + 'log.txt', 'w') as writer:
        writer.write("-start-\t")
        writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        writer.write('\n\n')

    print("\n-start- ", "(", datetime.datetime.now(), ")")
    
    torch.multiprocessing.set_start_method('spawn')
    train()

    print("\n-end- ", "(", datetime.datetime.now(), ")")

    with open(cfg.logdir + 'log.txt', 'a') as writer:
        writer.write('-end-\t')
        writer.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

