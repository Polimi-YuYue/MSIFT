# -*- coding:utf-8 -*-
"""
作者：于越
日期：2023年10月13日
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from models.gradient_reversal import revgrad
from models.networks import ConvNet, CrossTransformer_MOD_AVG


class MSIF(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super(MSIF, self).__init__()
        self.vib_cnn = ConvNet(dim)  # 振动 # shape (B,C,H,W)
        self.aco_cnn = ConvNet(dim)  # 声音
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), Rearrange('b c h w -> b (c h w)'))  # shape (B,C)
        # discriminator
        self.D = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 2))

        # fuse_transformer
        self.fuse_transformer = CrossTransformer_MOD_AVG(128, 3, 4, 32, 512, 0)  # shape (B,512)

        # multi_class classifier 在这里修改类别数
        self.fc_cls = nn.Sequential(nn.Linear(dim * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
                                    nn.Linear(128, 4))
        # binary classifier
        self.fc_binary = nn.Sequential(nn.Linear(dim * 4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(128, 2))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, vib, aco):
        # forward ConvNet
        vib_embeddings = self.vib_cnn(vib)  # 振动 shape (B,C,H,W)
        aco_embeddings = self.aco_cnn(aco)  # 声音 shape (B,C,H,W)
        # alpha = 2
        # reverse gradient
        alpha = torch.Tensor([2]).to(vib.device)
        vib_embedding_vec = revgrad(self.gap(vib_embeddings), alpha)  # 振动 shape (B,C)
        aco_embedding_vec = revgrad(self.gap(aco_embeddings), alpha)  # 声音 shape (B,C)
        # output: [B,2]
        D_vib_logits = self.D(vib_embedding_vec)
        D_aco_logits = self.D(aco_embedding_vec)

        # forward cross transformer
        vib_embeddings = rearrange(vib_embeddings, 'b c h w -> b (h w) c')
        aco_embeddings = rearrange(aco_embeddings, 'b c h w -> b (h w) c')
        output_pos = self.fuse_transformer(vib_embeddings, aco_embeddings)  # shape (B,512)
        output_logits = self.fc_cls(output_pos)
        binary_logits = self.fc_binary(output_pos)

        output0 = F.softmax(binary_logits, dim=-1)
        output1 = F.softmax(output_logits, dim=-1)
        output2 = torch.cat((output1[:, 0].unsqueeze(1), torch.sum(output1[:, 1:3], dim=1).unsqueeze(1)), 1)
        weight = 2 * torch.sigmoid(1 - F.cosine_similarity(output0, output2, dim=1))
        return binary_logits, output_logits, D_vib_logits, D_aco_logits, weight


if __name__ == "__main__":
    model = MSIF(128, 3, 4, 32, 512, 0)
    x0 = torch.randn(32, 3, 64, 64)
    x1 = torch.randn(32, 3, 64, 64)
    a, b, c = model(x0, x1)
    print(a.shape, b.shape, c.shape)
