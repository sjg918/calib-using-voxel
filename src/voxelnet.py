        
import torch
import spconv.pytorch as spconv
import torch.nn as nn

import torch.nn.functional as F
from src.losses import quaternion_distance


def FlattenFunc(sgmx, velox, sgm_local, velo_local):
    B = sgmx.batch_size
    sgmI = sgmx.indices
    veloI = velox.indices

    newsgm = torch.zeros((B, sgm_local, 1024), dtype=torch.float32, device=sgmx.features.device)
    newvelo = torch.zeros((B, velo_local, 1024), dtype=torch.float32, device=sgmx.features.device)

    for i in range(B):
        sgmmask = (sgmI[:, 0] == i)
        sgmtmp1 = sgmx.features[sgmmask]

        velomask = (veloI[:, 0] == i)
        velotmp1 = velox.features[velomask]

        if sgmtmp1.shape[0] >= sgm_local:
            sgmtmp1 = sgmtmp1[:sgm_local, :]
            newsgm[i, :sgm_local, :] = sgmtmp1
        else:
            numel = sgmmask.sum()
            newsgm[i, :numel, :] = sgmtmp1

        if velotmp1.shape[0] >= velo_local:
            velotmp1 = velotmp1[:velo_local, :]
            newvelo[i, :velo_local, :] = velotmp1
        else:
            numel = velomask.sum()
            newvelo[i, :numel, :] = velotmp1

        continue
    return newsgm, newvelo

    
class AttentionModule_ReLU(nn.Module):
    def __init__(self, in_channels, bias=True):
        super(AttentionModule_ReLU, self).__init__()
        self.linearq = nn.Sequential(
            nn.Linear(1024, in_channels, bias=bias),
            nn.LayerNorm(in_channels),
            nn.ReLU(),
        )
        self.lineark = nn.Sequential(
            nn.Linear(1024, in_channels, bias=bias),
            nn.LayerNorm(in_channels),
            nn.ReLU(),
        )
        self.linearv1 = nn.Sequential(
            nn.Linear(1024, in_channels//2, bias=bias),
            nn.LayerNorm(in_channels//2),
            nn.ReLU(),
        )
        self.linearv2 = nn.Sequential(
            nn.Linear(1024, in_channels//2, bias=bias),
            nn.LayerNorm(in_channels//2),
            nn.ReLU(),
        )


    def forward(self, sgm, velo):
        q = self.linearq(sgm)
        k = self.lineark(velo)
        attn = q @ k.permute(0, 2, 1).contiguous()
        attn = F.softmax(attn, dim=2)

        v1 = self.linearv1(velo)
        attn = attn @ v1

        v2 = self.linearv2(sgm)

        outf = torch.cat((v2, attn), dim=2)

        return outf
    

def mkSPConvNet_ReLU():
    return spconv.SparseSequential(
            spconv.SubMConv3d(3, 16, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            spconv.SparseConv3d(16, 24, 3, 2, padding=1, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            spconv.SubMConv3d(24, 24, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            spconv.SparseConv3d(24, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            spconv.SubMConv3d(32, 32, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            spconv.SparseConv3d(32, 64, 3, 2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SubMConv3d(64, 64, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            spconv.SparseConv3d(64, 128, 3, 2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spconv.SubMConv3d(128, 128, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spconv.SparseConv3d(128, 256, 3, 2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            spconv.SubMConv3d(256, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            spconv.SubMConv3d(256, 256, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            spconv.SparseConv3d(256, 512, 3, 2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            spconv.SubMConv3d(512, 512, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            spconv.SubMConv3d(512, 512, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            spconv.SubMConv3d(512, 512, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            spconv.SubMConv3d(512, 512, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            spconv.SparseConv3d(512, 1024, 3, 2, padding=1, bias=True), # 9
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            spconv.SubMConv3d(1024, 1024, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            spconv.SubMConv3d(1024, 1024, 3, 1, padding=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )


def mkMLP_ReLU(in_ch, out_ch, bias=True):
    return nn.Sequential(
            nn.Linear(in_ch, out_ch, bias=bias),
            nn.LayerNorm(out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch, bias=bias),
            nn.LayerNorm(out_ch),
            nn.ReLU(),
        )

    
class Deepv2_base(nn.Module):
    def __init__(self, sgm_local, velo_local, split=True, dropout=False, lambda1=1, lambda2=2):
        super(Deepv2_base, self).__init__()

        self.sgm_local = sgm_local 
        self.velo_local = velo_local
        self.split= split
        self.dropout = dropout
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.sgm_spconv = mkSPConvNet_ReLU()
        if split:
            self.velo_spconv = mkSPConvNet_ReLU()
        self.attn = AttentionModule_ReLU(1024, bias=False)
        self.linear3 = mkMLP_ReLU(1024 * sgm_local, 1024, bias=False)
        if dropout:
            self.droplast = nn.Dropout(p=0.2)
        self.r = nn.Linear(1024, 4)
        self.t = nn.Linear(1024, 3)


    def forward(self, input_sgm, input_velo, target=None):
        sgmx = self.sgm_spconv(input_sgm)
        if self.split:
            velox = self.velo_spconv(input_velo)
        else:
            velox = self.sgm_spconv(input_velo)

        B = sgmx.batch_size

        sgmx, velox = FlattenFunc(sgmx, velox, self.sgm_local, self.velo_local)
        x = self.attn(sgmx, velox) 
        x = x.view(B, -1).contiguous()
        x = self.linear3(x)
        if self.dropout and self.train:
            x = self.droplast(x)
        
        quaternion = self.r(x)
        quaternion = F.normalize(quaternion, dim=1)
        translate = self.t(x)
        
        if target is not None:
            rotloss = quaternion_distance(quaternion, target['quaternion']).mean()
            trsloss = F.smooth_l1_loss(translate, target['translate'], reduction='none').sum(dim=1).mean()
            loss = self.lambda1 * rotloss + self.lambda2  * trsloss
            return loss, rotloss, trsloss

        return quaternion, translate
    

class Deepv2_voxelnumtest(nn.Module):
    def __init__(self, sgm_local, velo_local, split=True, dropout=False):
        super(Deepv2_voxelnumtest, self).__init__()

        self.sgm_local = sgm_local 
        self.velo_local = velo_local
        self.split= split
        self.dropout = dropout
        
        self.sgm_spconv = mkSPConvNet_ReLU()
        if split:
            self.velo_spconv = mkSPConvNet_ReLU()


    def forward(self, input_sgm, input_velo, target=None):
        sgmx = self.sgm_spconv(input_sgm)
        if self.split:
            velox = self.velo_spconv(input_velo)
        else:
            velox = self.sgm_spconv(input_velo)
        return sgmx.features.shape[0], velox.features.shape[0]


class Deepv2_2cm(nn.Module):
    def __init__(self, sgm_local, velo_local, split=True, dropout=False):
        super(Deepv2_2cm, self).__init__()

        self.sgm_local = sgm_local 
        self.velo_local = velo_local
        self.split= split
        self.dropout = dropout
        
        self.sgm_spconv = mkSPConvNet_ReLU()
        if split:
            self.velo_spconv = mkSPConvNet_ReLU()
        self.attn = AttentionModule_ReLU(128, bias=False)
        self.linear3 = mkMLP_ReLU(128 * sgm_local, 1024, bias=False)
        if dropout:
            self.droplast = nn.Dropout(p=0.2)
        self.r = nn.Linear(1024, 4)
        self.t = nn.Linear(1024, 3)


    def forward(self, input_sgm, input_velo, target=None):
        sgmx = self.sgm_spconv(input_sgm)
        if self.split:
            velox = self.velo_spconv(input_velo)
        else:
            velox = self.sgm_spconv(input_velo)
        #return sgmx.features.shape[0], velox.features.shape[0]

        B = sgmx.batch_size

        sgmx, velox = FlattenFunc(sgmx, velox, self.sgm_local, self.velo_local)
        x = self.attn(sgmx, velox) 
        x = x.view(B, -1).contiguous()
        x = self.linear3(x)
        if self.dropout and self.train:
            x = self.droplast(x)
        
        quaternion = self.r(x)
        quaternion = F.normalize(quaternion, dim=1)
        translate = self.t(x)
        
        if target is not None:
            rotloss = quaternion_distance(quaternion, target['quaternion']).mean()
            trsloss = F.smooth_l1_loss(translate, target['translate'], reduction='none').sum(dim=1).mean()
            loss = rotloss + 2.0 * trsloss
            return loss, rotloss, trsloss

        return quaternion, translate
