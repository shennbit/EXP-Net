import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

import copy
import math
import logging
import numpy as np
from einops import rearrange
from os.path import join as pjoin


class VisionTransformer_UNET(nn.Module):
    def __init__(self, config):
        super(VisionTransformer_UNET, self).__init__()
        self.config = config
        self.encode_transformer = EncodeTransformer(self.config)
        self.decoder_with_cross_attention = DecoderWithCrossAttention(self.config)
        self.segmentation_head = SegmentationHead(self.config)

    def forward(self, x):
        x = self.encode_transformer(x)
        x, y = self.decoder_with_cross_attention(x)
        out = self.segmentation_head(x)
        return out, y


class EncodeTransformer(nn.Module):
    def __init__(self, config):
        super(EncodeTransformer, self).__init__()
        self.config = config

        # l1
        self.resblock_l1_1 = ResidualBasicBlock(self.config.in_chan, self.config.base_chan, stride=1)
        self.resblock_l1_2 = ResidualBasicBlock(self.config.base_chan, self.config.base_chan, stride=1)

        # l2
        self.maxpool_l2 = nn.MaxPool2d(2)
        self.resblock_l2 = ResidualBasicBlock(self.config.base_chan, 2*self.config.base_chan, stride=1)
        #self.transblock_l2 = TransBasicBlock(self.config, dim=2*self.config.base_chan, dim_head=2*self.config.base_chan/4)

        # l3
        self.maxpool_l3 = nn.MaxPool2d(2)
        self.resblock_l3 = ResidualBasicBlock(2*self.config.base_chan, 4*self.config.base_chan, stride=1)
        #self.transblock_l3 = TransBasicBlock(self.config, dim=4*self.config.base_chan, dim_head=4*self.config.base_chan/4)

        # l4
        self.maxpool_l4 = nn.MaxPool2d(2)
        self.resblock_l4 = ResidualBasicBlock(4*self.config.base_chan, 8*self.config.base_chan, stride=1)
        #self.transblock_l4 = TransBasicBlock(self.config, dim=8*self.config.base_chan, dim_head=8*self.config.base_chan/4)

        # l5
        self.maxpool_l5 = nn.MaxPool2d(2)
        self.resblock_l5 = ResidualBasicBlock(8*self.config.base_chan, 16*self.config.base_chan, stride=1)
        self.transblock_l5 = TransBasicBlock(self.config, dim=16*self.config.base_chan, dim_head=16*self.config.base_chan/4)

    def forward(self, x):
        features = []
        # l1
        # x: [32,1,480,640]
        x1 = self.resblock_l1_1(x) # x1: [32,32,480,640]
        x1 = self.resblock_l1_2(x1) # x1: [32,32,480,640]
        features.append(x1)

        # l2
        x2 = self.maxpool_l2(x1) # x2: [32,32,240,320]
        x2 = self.resblock_l2(x2) # x2: [32,64,240,320]
        #x2 = self.transblock_l2(x2) # x2: [32,64,240,320]
        features.append(x2)

        # l3
        x3 = self.maxpool_l3(x2) # x3: [32,64,120,160]
        x3 = self.resblock_l3(x3) # x3: [32,128,120,160]
        #x3 = self.transblock_l3(x3) # x3: [32,128,120,160]
        features.append(x3)

        # l4
        x4 = self.maxpool_l4(x3) # x4: [32,128,60,80]
        x4 = self.resblock_l4(x4) # x4: [32,256,60,80]
        #x4 = self.transblock_l4(x4) # x4: [32,256,60,80]
        features.append(x4)

        # l5
        x5 = self.maxpool_l5(x4) # x5: [32,256,30,40]
        x5 = self.resblock_l5(x5) # x5: [32,512,30,40]
        x5 = self.transblock_l5(x5) # x5: [32,512,30,40]
        features.append(x5)

        return features #[x1, x2, x3, x4, x5]


class DecoderWithCrossAttention(nn.Module):
    def __init__(self, config):
        super(DecoderWithCrossAttention, self).__init__()
        self.config = config

        # l4
        self.conv_l4_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16*self.config.base_chan, 8*self.config.base_chan, kernel_size=1),
            nn.BatchNorm2d(8*self.config.base_chan), nn.ReLU())  # inplace=True
        self.conv_l4_2 = ResidualBasicBlock(16*self.config.base_chan, 8*self.config.base_chan, stride=1)

        # self.crossblock_l4 = TransCrossBlock(self.config, in_ch=16*self.config.base_chan, out_ch=8*self.config.base_chan,
        #                                      dim_head=8*self.config.base_chan/4)
        # self.conv_l4 = ResidualBasicBlock(8 * self.config.base_chan, 8 * self.config.base_chan, stride=1)
            # nn.Sequential(
            # nn.Conv2d(8*self.config.base_chan, 8*self.config.base_chan,kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(8*self.config.base_chan),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(8*self.config.base_chan,8*self.config.base_chan,kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(8*self.config.base_chan),
            # nn.ReLU(inplace=True))

        # l3
        self.conv_l3_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(8*self.config.base_chan, 4*self.config.base_chan, kernel_size=1),
            nn.BatchNorm2d(4*self.config.base_chan), nn.ReLU())  # inplace=True
        self.conv_l3_2 = ResidualBasicBlock(8*self.config.base_chan, 4*self.config.base_chan, stride=1)
        # self.crossblock_l3 = TransCrossBlock(self.config, in_ch=8*self.config.base_chan, out_ch=4*self.config.base_chan,
        #                                      dim_head=4*self.config.base_chan/4)
        # self.conv_l3 = ResidualBasicBlock(4 * self.config.base_chan, 4 * self.config.base_chan, stride=1)
            # nn.Sequential(
            # nn.Conv2d(4*self.config.base_chan, 4*self.config.base_chan, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(4*self.config.base_chan),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(4*self.config.base_chan, 4*self.config.base_chan, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(4*self.config.base_chan),
            # nn.ReLU(inplace=True))

        # l2
        self.conv_l2_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(4*self.config.base_chan, 2*self.config.base_chan, kernel_size=1),
            nn.BatchNorm2d(2*self.config.base_chan), nn.ReLU())  # inplace=True
        self.conv_l2_2 = ResidualBasicBlock(4*self.config.base_chan, 2*self.config.base_chan, stride=1)
        # self.crossblock_l2 = TransCrossBlock(self.config, in_ch=4*self.config.base_chan, out_ch=2*self.config.base_chan,
        #                                      dim_head=2*self.config.base_chan/4)
        # self.conv_l2 = ResidualBasicBlock(2 * self.config.base_chan, 2 * self.config.base_chan, stride=1)
            # nn.Sequential(
            # nn.Conv2d(2*self.config.base_chan, 2*self.config.base_chan, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(2*self.config.base_chan),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(2*self.config.base_chan, 2*self.config.base_chan, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(2*self.config.base_chan),
            # nn.ReLU(inplace=True))

        # l1
        self.conv_l1_1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(2*self.config.base_chan, self.config.base_chan, kernel_size=1),
            nn.BatchNorm2d(self.config.base_chan), nn.ReLU()) #inplace=True
        self.conv_l1_2 = ResidualBasicBlock(2*self.config.base_chan, self.config.base_chan, stride=1)

    def forward(self, features):
        required_features = []
        # l4
        required_features.append(features[4])
        x4 = self.conv_l4_1(features[4])
        x4 = torch.cat([x4, features[3]], dim=1)
        x4 = self.conv_l4_2(x4)
        required_features.append(x4)
        # x4 = self.crossblock_l4(features[4], features[3])
        # x4 = self.conv_l4(x4)
        #
        # l3
        x3 = self.conv_l3_1(x4)
        x3 = torch.cat([x3, features[2]], dim=1)
        x3 = self.conv_l3_2(x3)
        required_features.append(x3)
        # x3 = self.crossblock_l3(x4, features[2])
        # x3 = self.conv_l3(x3)
        #
        # l2
        x2 = self.conv_l2_1(x3)
        x2 = torch.cat([x2, features[1]], dim=1)
        x2 = self.conv_l2_2(x2)
        required_features.append(x2)
        # x2 = self.crossblock_l2(x3, features[1])
        # x2 = self.conv_l2(x2)

        # l1
        x1 = self.conv_l1_1(x2)
        out = torch.cat([x1, features[0]], dim=1)
        out = self.conv_l1_2(out)
        required_features.append(out)

        return out, required_features


class SegmentationHead(nn.Module):
    def __init__(self, config):
        super(SegmentationHead, self).__init__()
        self.config = config
        self.conv = nn.Conv2d(self.config.base_chan, self.config.out_chan, kernel_size=1)

    def forward(self, x):
        out = self.conv(x)

        return out


#EncodeTransformer
class ResidualBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResidualBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU()

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_planes),
                nn.ReLU())
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residue = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)


        out += self.shortcut(residue)

        return out


class TransBasicBlock(nn.Module):
    def __init__(self, config, dim, dim_head):
        super(TransBasicBlock, self).__init__()
        self.config = config
        self.dim = dim
        self.dim_head = dim_head

        self.bn1 = nn.BatchNorm2d(self.dim)
        self.attn = LinearAttention(self.dim, self.dim_head, reduce_size=self.config.reduce_size)

        self.bn2 = nn.BatchNorm2d(self.dim)
        self.mlp = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        # conv1x1 has not difference with mlp in performance

    def forward(self, x):
        out = self.bn1(x)
        out = self.attn(out)

        out = out + x

        residue = out
        out = self.bn2(out)
        out = self.mlp(out)
        out = self.relu(out)

        out += residue

        return out


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head, reduce_size, attn_drop=0.1, proj_drop=0.1, heads=4):
        super(LinearAttention, self).__init__()

        self.inner_dim = int(dim_head * heads)
        self.heads = int(heads)
        self.scale = dim_head ** (-0.5)
        self.dim_head = int(dim_head)
        self.reduce_size = reduce_size
        self.dim = int(dim)

        self.to_qkv = depthwise_separable_conv(self.dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, self.dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pe = PositionalEncodingPermute2D(self.dim)

    def forward(self, x):
        B, C, H, W = x.shape

        pe = self.pe(x)
        x = x + pe

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size[0], w=self.reduce_size[1]), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out


class TransCrossBlock(nn.Module):
    def __init__(self, config, in_ch, out_ch, dim_head):
        super(TransCrossBlock, self).__init__()

        self.config = config
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.dim_head = dim_head

        self.bn_l = nn.BatchNorm2d(self.in_ch)
        self.bn_h = nn.BatchNorm2d(self.out_ch)

        self.conv_ch = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1)
        self.attn = LinearAttentionDecoder(in_ch=self.in_ch, out_ch=self.out_ch, dim_head=self.dim_head,
                                           reduce_size=self.config.reduce_size)

        self.bn2 = nn.BatchNorm2d(self.out_ch)
        self.mlp = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, q):
        residue = F.interpolate(self.conv_ch(x), size=q.shape[-2:], mode='bilinear', align_corners=True)

        x = self.bn_l(x)
        q = self.bn_h(q)
        out= self.attn(x, q)

        out += residue

        residue = out
        out = self.bn2(out)
        out = self.mlp(out)
        out = self.relu(out)

        out += residue

        return out


class LinearAttentionDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, dim_head, reduce_size, attn_drop=0.1, proj_drop=0.1, heads=4):
        super(LinearAttentionDecoder, self).__init__()

        self.inner_dim = int(dim_head * heads)
        self.heads = int(heads)
        self.scale = dim_head ** (-0.5)
        self.dim_head = int(dim_head)
        self.reduce_size = reduce_size
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)

        self.to_kv = depthwise_separable_conv(self.in_ch, self.inner_dim * 2)
        self.to_q = depthwise_separable_conv(self.out_ch, self.inner_dim)
        self.to_out = depthwise_separable_conv(self.inner_dim, self.out_ch)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.Xpe = PositionalEncodingPermute2D(2*self.out_ch)
        self.Qpe = PositionalEncodingPermute2D(self.out_ch)

    def forward(self, x, q):
        #S=q Y=x
        BX, CX, HX, WX = x.shape
        BQ, CQ, HQ, WQ = q.shape

        xpe = self.Xpe(x)
        x = x + xpe
        k, v = self.to_kv(x).chunk(2, dim=1)  # B, inner_dim, H, W

        qpe = self.Qpe(q)
        q = q + qpe
        q = self.to_q(q)  # BH, inner_dim, HH, WH

        k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=HQ, w=WQ)

        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size[0], w=self.reduce_size[1]), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=HQ, w=WQ, dim_head=self.dim_head,
                        heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias, stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)