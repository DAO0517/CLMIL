"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, memory_queue=None, label_queue=None,mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            label_queue = label_queue.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, label_queue.T).to(device)  # (bsz,bsz)
            inverse_mask = torch.logical_not(mask)
        else:
            mask = mask.float().to(device)


        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, memory_queue),  # 查询的feature与所有的feature对比
            self.temperature)  # torch.matmul的结果为(bsz,bsz*n_view)或者(bsz*n_view,bsz*n_view)torch.div(input,other)直接除以一个数字，相似度
        # for numerical stability
        # 沿着第一个维度取最大值，logits_max 是每行的最大值，_ 则是最大值的索引。这一步是为了数值稳定性，通过减去最大值，避免了指数爆炸问题
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # 每行的最大值
        logits = anchor_dot_contrast - logits_max.detach()  # 计算相似度减去最大值

        # compute log_prob
        pos_logits = torch.exp(logits) * mask  # 对角线为0
        neg_logits = torch.exp(logits) * inverse_mask
        log_prob = torch.log(pos_logits.sum(1, keepdim=True)+neg_logits.sum(1, keepdim=True)) - torch.log(pos_logits.sum(1, keepdim=True))  # 计算了对数概率。首先，对数概率被计算为 logits 减去 log(exp(logits) 的和。这样处理可以帮助数值稳定性

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)  # 算了每个样本中正对比的数量，避免了除以零的情况。
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(1, batch_size).mean()
        loss = loss.mean()

        return loss
