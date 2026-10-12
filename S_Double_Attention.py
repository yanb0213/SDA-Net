
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentAwareRelativeBias(nn.Module):
    def __init__(self, num_heads, spatial_size=(8, 8), content_channels=512):
        super().__init__()
        self.num_heads = num_heads
        max_dist = max(spatial_size)
        self.max_dist = max_dist
        self.base_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_dist - 1, 2 * max_dist - 1))
        nn.init.normal_(self.base_bias, std=0.02)
        self.modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(content_channels, content_channels // 8),
            nn.ReLU(),
            nn.Linear(content_channels // 8, 1),
            nn.Sigmoid()
        )

    def _get_relative_position_bias(self, H, W, device, scale):
        grid_h = torch.arange(H, device=device)
        grid_w = torch.arange(W, device=device)
        grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing='ij')
        pos = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=1)
        rel_diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        rel_h = rel_diff[..., 0] + self.max_dist - 1
        rel_w = rel_diff[..., 1] + self.max_dist - 1
        rel_h = rel_h.clamp(0, 2 * self.max_dist - 2)
        rel_w = rel_w.clamp(0, 2 * self.max_dist - 2)

        base = self.base_bias[:, rel_h.long(), rel_w.long()]
        base = base.unsqueeze(0)
        return base * scale.view(-1, 1, 1, 1)

    def forward(self, H, W, content_feat, device):
        scale = self.modulator(content_feat)  # (B, 1)
        bias = self._get_relative_position_bias(H, W, device, scale)
        return bias


class StrokeCrossRegionAttention(nn.Module):
    """
    与论文 3.3.1 节严格对应：
    s_j^(h) = k_j^(h) · p_h / sqrt(d_h)
    k?_j^(h) = k_j^(h) ⊙ σ(s_j^(h))
    S_ij^(h) = q_i^(h) · k?_j^(h) / sqrt(d_h) + b_ij^(h)
    """

    def __init__(self, embed_dim, num_heads=8, spatial_size=(8, 8)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.cross_q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])
        self.non_cross_k_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])
        self.original_v_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])

        self.prototype = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)

        self.content_bias = ContentAwareRelativeBias(num_heads, spatial_size, embed_dim)

    def forward(self, non_cross_feat, cross_feat, original_feat):
        B, C, H, W = non_cross_feat.shape

        Q = torch.cat([q(cross_feat) for q in self.cross_q_heads], dim=1)
        K = torch.cat([k(non_cross_feat) for k in self.non_cross_k_heads], dim=1)
        V = torch.cat([v(original_feat) for v in self.original_v_heads], dim=1)

        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)  # (B, heads, L, d)
        K = K.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)
        V = V.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)

        s = torch.einsum('bhld,hd->bhl', K, self.prototype) / (self.head_dim ** 0.5)  # (B, heads, L)
        gate = torch.sigmoid(s).unsqueeze(-1)  # (B, heads, L, 1)
        K_mod = K * gate                        # (B, heads, L, d) 即 k?_j^(h)

        dynamic_bias = self.content_bias(H, W, cross_feat, Q.device)

        attn_score = torch.matmul(Q, K_mod.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_score = attn_score + dynamic_bias

        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = F.dropout(attn_weight, 0.05, training=self.training)
        out = torch.matmul(attn_weight, V)

        self.attn_weight = attn_weight

        out = out.transpose(2, 3).reshape(B, C, H, W)
        return out


class StrokeSegmentComponentAttention(nn.Module):
    

    def __init__(self, embed_dim, num_heads=8, spatial_size=(8, 8)):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.non_cross_q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])
        self.non_cross_k_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])
        self.ssra_v_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.Conv2d(embed_dim, self.head_dim, 1),
                nn.Dropout2d(0.05)
            ) for _ in range(num_heads)
        ])


        self.content_bias = ContentAwareRelativeBias(num_heads, spatial_size, embed_dim)

    def forward(self, non_cross_feat, original_feat, ssra_feat, shared_prototype):
        
        B, C, H, W = non_cross_feat.shape

        Q = torch.cat([q(original_feat) for q in self.non_cross_q_heads], dim=1)
        K = torch.cat([k(original_feat) for k in self.non_cross_k_heads], dim=1)
        V = torch.cat([v(ssra_feat) for v in self.ssra_v_heads], dim=1)

        Q = Q.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)
        K = K.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)
        V = V.reshape(B, self.num_heads, self.head_dim, H, W).flatten(3).transpose(2, 3)

        s = torch.einsum('bhld,hd->bhl', K, shared_prototype) / (self.head_dim ** 0.5)
        gate = torch.sigmoid(s).unsqueeze(-1)
        K_mod = K * gate

        dynamic_bias = self.content_bias(H, W, original_feat, Q.device)

        attn_score = torch.matmul(Q, K_mod.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_score = attn_score + dynamic_bias

        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = F.dropout(attn_weight, 0.05, training=self.training)
        out = torch.matmul(attn_weight, V)

        self.attn_weight = attn_weight

        out = out.transpose(2, 3).reshape(B, C, H, W)
        return out


class SDoubleAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads=8, spatial_size=(8, 8)):
        super().__init__()
        self.channel_adjust = nn.Conv2d(input_dim, embed_dim, kernel_size=1)
        self.ssra = StrokeCrossRegionAttention(embed_dim, num_heads, spatial_size)
        self.ssca = StrokeSegmentComponentAttention(embed_dim, num_heads, spatial_size)

    def forward(self, cross_feat, non_cross_feat, original_feat):
        cross_feat = self.channel_adjust(cross_feat)
        non_cross_feat = self.channel_adjust(non_cross_feat)
        original_feat = self.channel_adjust(original_feat)

        ssra_out = self.ssra(non_cross_feat, cross_feat, original_feat)

        ssca_out = self.ssca(non_cross_feat, original_feat, ssra_out, self.ssra.prototype)

        return ssca_out
