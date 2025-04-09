# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
from mmengine.model import BaseModule
from typing import List, Optional, Tuple, Union
from torch import Tensor

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataAnomalySample

import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def channel_wise_square_sum(feature_map):
    """
    计算特征图的通道维度平方和。

    Args:
        feature_map (Tensor): 输入特征图，尺寸为 (B, C, H, W)

    Returns:
        Tensor: 沿着通道维度的平方和，尺寸为 (B, 1, H, W)
    """
    # 对特征图的每个通道进行平方
    squared_feature_map = feature_map ** 2

    # 沿着通道维度求和 (dim=1 表示对通道进行求和)
    sum_squared = torch.sum(squared_feature_map, dim=1, keepdim=True)

    return sum_squared


@MODELS.register_module()
class DistillHead(BaseModule):
    """Head for SimMIM Pre-training.

    Args:
        patch_size (int): Patch size of each token.
        loss (dict): The config for loss.
    """

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss_module = MODELS.build(loss)

    def _stack_batch_gt(self, batch_data_samples) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, feats: torch.Tensor, data_samples: List[DataAnomalySample], **kwargs) -> torch.Tensor:
        """Generate loss.

        This method will expand mask to the size of the original image.

        Args:
            pred (torch.Tensor): The reconstructed image (B, C, H, W).
            target (torch.Tensor): The target image (B, C, H, W).
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        feats = feats[0]  # 返回的是tuple, size=1
        anomaly_maps = self._stack_batch_gt(data_samples)
        #print('distill head anomaly_maps ', anomaly_maps.size())

        #print('distill feats ', feats.size())  # [bs, 1025, 7, 7]
        bs = feats.shape[0]
        feats_sum = channel_wise_square_sum(feature_map=feats)  # [bs, 1, 7, 7]   # 获取特征激活图，2次方能够更加凸显高亮区域
        #print('distill feats_sum ', feats_sum.size())
        loss = dict()
        anomaly_maps = resize(
            input=anomaly_maps,
            size=feats.shape[2:],
            mode='bilinear',
            align_corners=False)  # [b, 7, 7]
        #print('distill head anomaly_maps resize ', anomaly_maps.size())

        losses = dict()
        loss = self.loss_module(feats_sum.contiguous().view(bs, -1), anomaly_maps.contiguous().view(bs, -1))
        losses['loss'] = loss

        return losses


@MODELS.register_module()
class MultiLayerDistillHead(BaseModule):
    """Head for SimMIM Pre-training.

    Args:
        patch_size (int): Patch size of each token.
        loss (dict): The config for loss.
    """

    def __init__(self, loss: dict) -> None:
        super().__init__()
        self.loss_module = MODELS.build(loss)

    def _stack_batch_gt(self, batch_data_samples) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, feats_tuple: torch.Tensor, data_samples: List[DataAnomalySample], **kwargs) -> torch.Tensor:
        """Generate loss.

        This method will expand mask to the size of the original image.

        Args:
            pred (torch.Tensor): The reconstructed image (B, C, H, W).
            target (torch.Tensor): The target image (B, C, H, W).
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        #feats = feats[0]  # 返回的是tuple, size=1
        anomaly_maps = self._stack_batch_gt(data_samples)
        #print('distill head anomaly_maps ', anomaly_maps.size())

        #print('distill feats ', len(feats_tuple), ' ', feats_tuple[0].size())  # [bs, 1024, 7, 7]
        losses = dict()
        loss = 0.0
        for feats in feats_tuple:
            bs = feats.shape[0]
            feats_sum = channel_wise_square_sum(feature_map=feats)  # [bs, 1, 7, 7]   # 获取特征激活图，2次方能够更加凸显高亮区域
            #print('distill feats_sum ', feats_sum.size())
        
            anomaly_maps = resize(
                input=anomaly_maps,
                size=feats.shape[2:],
                mode='bilinear',
                align_corners=False)  # [b, 7, 7]
            #print('distill head anomaly_maps resize ', anomaly_maps.size())

            loss += self.loss_module(feats_sum.contiguous().view(bs, -1), anomaly_maps.contiguous().view(bs, -1))
        losses['loss'] = loss

        return losses


import torch
import torch.nn as nn

class AdapterMLPWithAttention(BaseModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        
        # MLP部分
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 第一层
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)      # 输出层
        )
        
        # 7x7的多头注意力块
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=1)

        self.output_size = output_size
        self.layer_norm_attn = nn.LayerNorm(input_size)
        
    def forward(self, x):
        # 输入形状: [bs, 1024, 7, 7]
        bs, c, h, w = x.shape
        #print('init x ', x.size())
        
        # 将输入reshape为适合注意力的形状: [7*7, bs, 1024]
        x = x.view(bs, c, h * w).permute(2, 0, 1)  # [49, bs, 1024]
        #print('x ', x.size())
        
        # 通过注意力层
        attn_output, _ = self.attention(x, x, x)  # 自注意力
        attn_output = self.layer_norm_attn(attn_output)  # 归一化
        #print('attn_output ', attn_output.size())
        attn_output = attn_output.permute(1, 2, 0).view(bs, c, h, w)  # [bs, 1024, 7, 7]
        #print('attn_output 2 ', attn_output.size())
        
        # 经过MLP层
        x = attn_output.view(bs, c, -1).transpose(1, 2)  # Reshape to [bs, 49, 1024]
        #print('x 2 ', x.size())
        x = self.mlp(x)  # Apply MLP: [bs, 49, 1]
        #print('x 3 ', x.size())
        x = x.transpose(1, 2)  # [bs, 1, 49]
        #print('x 4 ', x.size())
        output = x.view(bs, self.output_size, h, w)  # [bs, 1, 7, 7]
        #print('output ', output.size())
        
        return output


@MODELS.register_module()
class MultiLayerDistillWithAttnHead(BaseModule):
    """Head for SimMIM Pre-training.

    Args:
        patch_size (int): Patch size of each token.
        loss (dict): The config for loss.
    """

    def __init__(self, loss: dict, input_size: int) -> None:
        super().__init__()
        self.loss_module = MODELS.build(loss)
        self.adaptor =  AdapterMLPWithAttention(input_size, 512, 1)

    def _stack_batch_gt(self, batch_data_samples) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss(self, feats_tuple: torch.Tensor, data_samples: List[DataAnomalySample], **kwargs) -> torch.Tensor:
        """Generate loss.

        This method will expand mask to the size of the original image.

        Args:
            pred (torch.Tensor): The reconstructed image (B, C, H, W).
            target (torch.Tensor): The target image (B, C, H, W).
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        #feats = feats[0]  # 返回的是tuple, size=1
        anomaly_maps = self._stack_batch_gt(data_samples)
        #print('distill head anomaly_maps ', anomaly_maps.size())

        #print('distill feats ', len(feats_tuple), ' ', feats_tuple[0].size())  # [bs, 1024, 7, 7]
        losses = dict()
        loss = 0.0
        for feats in feats_tuple:
            bs = feats.shape[0]
            #feats_sum = channel_wise_square_sum(feature_map=feats)  # [bs, 1, 7, 7]   # 获取特征激活图，2次方能够更加凸显高亮区域
            feats_sum = self.adaptor(feats)
            #print('distill feats_sum ', feats_sum.size())
        
            anomaly_maps = resize(
                input=anomaly_maps,
                size=feats.shape[2:],
                mode='bilinear',
                align_corners=False)  # [b, 7, 7]
            #print('distill head anomaly_maps resize ', anomaly_maps.size())

            loss += self.loss_module(feats_sum.contiguous().view(bs, -1), anomaly_maps.contiguous().view(bs, -1))
        losses['loss'] = loss

        return losses
