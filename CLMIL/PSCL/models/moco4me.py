import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MoCo4me(nn.Module):
    def __init__(self, arch='resnet50', feature_dim=128, moco_momentum=0.999, num_class=3, mlp=True):
        super(MoCo4me, self).__init__()
        
        self.m = moco_momentum                
        
        self.encoder_q = models.__dict__[arch](num_classes=feature_dim)
        self.encoder_k = models.__dict__[arch](num_classes=feature_dim)

                
        if mlp:  # 如果mlp为true，需要对模型最后一层进行修改
            dim_mlp = self.encoder_q.fc.weight.shape[1]  # 获取最后一层的输入维度（即全连接层的权重矩阵的列数）
            # nn.Sequential将这些组合起来
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            
        for (param_q, param_k) in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # 这样在模型的训练过程中，这些参数不会被更新，固定参数

        
    def forward(self, im_q, im_k):
        query = self.encoder_q(im_q)

        query = nn.functional.normalize(query, dim=1)
        
        with torch.no_grad():
            for (param_eq, param_ek) in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_ek.data.copy_(param_ek.data * self.m + param_eq.data * (1. - self.m))
            
            # shuffle BN
            idx_shuffle = torch.randperm(im_k.size(0)).cuda()  # 生成一个随机排列的索引idx_shuffle，用于对im_k进行洗牌操作
            idx_unshuffle = torch.argsort(idx_shuffle)  # 根据idx_shuffle对其进行排序，得到原始顺序的索引idx_unshuffle
            
            key = self.encoder_k(im_k[idx_shuffle])

            key = nn.functional.normalize(key, dim=1)  # 对key进行归一化操作
            key = key[idx_unshuffle]  # 根据idx_unshuffle恢复其原始顺序
            
        return query, key

